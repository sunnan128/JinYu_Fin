# ── LexAI RAGAS 自动化评估流水线 ──
#
# 面试重点:
# 1. RAGAS 是工业界主流的 RAG 评估框架（faithfulness/answer_relevancy 等标准指标）
# 2. 量化评估是"优化有据可依"的关键——没有评估的优化就是玄学
# 3. 支持 A/B 对比：同一数据集、不同配置、可复现的量化结果
# 4. LLM-as-a-Judge 兜底：RAGAS 不可用时仍能产出可用指标
#
# 用法:
#   python -m eval.run_eval --mode baseline   # 基础模式（无 Rerank）
#   python -m eval.run_eval --mode rerank      # Rerank 模式
#   python -m eval.run_eval --mode both        # 两种模式都跑（默认）
#
# 输出:
#   eval/results/baseline_metrics.json
#   eval/results/rerank_metrics.json

import json
import os
import sys
import time
import argparse
import requests
import socket
from typing import List, Dict, Any, Optional

# ── 编码修复：防止 Windows GBK 控制台无法输出 emoji 导致崩溃 ──
if sys.platform == "win32" and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── 配置 ──
_DEFAULT_PORT = 8006  # 与 backend/config.py API_PORT = 8006 保持一致
try:
    from backend.config import settings as _app_config
    API_URL = f"http://localhost:{_app_config.API_PORT}"
except (ImportError, AttributeError):
    # config 导入失败时（如缺少依赖），回退到默认端口 8006
    API_URL = f"http://localhost:{_DEFAULT_PORT}"
    print(f"  [INFO] 使用默认端口 {_DEFAULT_PORT}（backend.config 导入失败）")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── DeepSeek Judge ──
# 将 DeepSeek 配置注入环境变量，供 RAGAS / langchain 读取
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from backend.config import settings
    _DEEPSEEK_API_KEY = settings.OPENAI_API_KEY
    _DEEPSEEK_BASE_URL = settings.OPENAI_BASE_URL
    _DEEPSEEK_MODEL = settings.LLM_MODEL or "deepseek-chat"
except (ImportError, AttributeError):
    # DeepSeek 配置导入失败（如 backend.config 依赖缺失），从环境变量读取
    _DEEPSEEK_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    _DEEPSEEK_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    _DEEPSEEK_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

# 设置环境变量（RAGAS 内部使用 langchain 读取）
os.environ["OPENAI_API_KEY"] = _DEEPSEEK_API_KEY
os.environ["OPENAI_BASE_URL"] = _DEEPSEEK_BASE_URL
os.environ["OPENAI_MODEL"] = _DEEPSEEK_MODEL

# ── 初始化 RAGAS 所需的 LLM 和 Embeddings（延迟初始化，避免导入时阻塞） ──
# RAGAS 0.4.x 的 metrics 构造器需要 LangChain ChatModel 和 Embeddings 实例
_ragas_llm = None
_ragas_embeddings = None

def _init_ragas_llm():
    """延迟初始化 RAGAS LLM（兼容 LangChain BaseChatModel 接口）

    使用 langchain_core 的 BaseChatModel 创建一个简单的 DeepSeek 适配器。
    避免导入 langchain_openai 或 langchain_community 导致的额外依赖和网络阻塞。
    """
    global _ragas_llm
    if _ragas_llm is not None:
        return _ragas_llm

    import requests as _requests
    from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    from typing import Any, Iterator

    class _DeepSeekChat(BaseChatModel):
        """最简 DeepSeek ChatModel 适配器，供 RAGAS 评估使用"""
        model_name: str = _DEEPSEEK_MODEL
        api_key: str = _DEEPSEEK_API_KEY
        base_url: str = _DEEPSEEK_BASE_URL
        temperature: float = 0.1

        @property
        def _llm_type(self) -> str:
            return "deepseek-ragas"

        def _generate(
            self,
            messages: list,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            msgs = []
            for m in messages:
                if isinstance(m, HumanMessage):
                    msgs.append({"role": "user", "content": m.content})
                elif isinstance(m, AIMessage):
                    msgs.append({"role": "assistant", "content": m.content})
                elif isinstance(m, BaseMessage):
                    msgs.append({"role": "system", "content": m.content})
                elif isinstance(m, dict):
                    msgs.append(m)
                else:
                    msgs.append({"role": "user", "content": str(m)})

            resp = _requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": msgs,
                    "temperature": self.temperature,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data['choices'][0]['message']['content']
            return ChatResult(
                generations=[ChatGeneration(
                    message=AIMessage(content=content)
                )]
            )

    try:
        _ragas_llm = _DeepSeekChat()
        print(f"  RAGAS LLM: {_DEEPSEEK_MODEL} @ {_DEEPSEEK_BASE_URL}")
        return _ragas_llm
    except Exception as e:
        print(f"  [WARN] RAGAS LLM 初始化失败: {e}")
        return None

def _init_ragas_embeddings():
    """延迟初始化 RAGAS Embeddings（本地 bge 模型）"""
    global _ragas_embeddings
    if _ragas_embeddings is not None:
        return _ragas_embeddings
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        _ragas_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _ragas_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
    print(f"  RAGAS Embeddings: BAAI/bge-large-zh-v1.5")
    return _ragas_embeddings

# ── RAGAS 可选导入（含版本检测与多路径兼容） ──
# RAGAS 0.4.x 中 metrics 导入路径已废弃，需从 ragas.metrics.collections 导入
# 我们尝试两种导入路径，确保兼容不同的 RAGAS 版本

def _get_pkg_version(pkg_name: str) -> str:
    """获取已安装包的版本，未安装时返回 'not installed'"""
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg_name)
    except Exception:
        return "not installed"


def _patch_vertexai_for_ragas():
    """为 RAGAS 0.4.x 修补缺失的 langchain_community.chat_models.vertexai 模块
    
    RAGAS 0.4.3 的 metrics 内部依赖 langchain_community，而 langchain-community>=0.4.0
    移除了 chat_models.vertexai 子模块（惰性加载失败）。
    
    此函数创建一个轻量 stub 模块注入 sys.modules，使 RAGAS 的 metrics 可以正常导入，
    而不影响其他功能（本系统不使用 Vertex AI）。
    """
    import sys
    if 'langchain_community.chat_models.vertexai' in sys.modules:
        return True  # 已存在，不需要修补

    try:
        # 尝试正常导入
        import langchain_community.chat_models.vertexai
        return True
    except (ImportError, ModuleNotFoundError):
        pass

    # 创建 stub 模块
    from types import ModuleType
    stub = ModuleType('langchain_community.chat_models.vertexai')
    stub.__doc__ = 'Stub module for RAGAS compatibility (Vertex AI not used)'
    # 提供 RAGAS 期望存在的最小接口
    try:
        from langchain_core.language_models.chat_models import BaseChatModel as _BaseChatModel
    except ImportError:
        class _BaseChatModel:
            pass

    class ChatVertexAI(_BaseChatModel):
        """Stub: 占位实现，不执行任何 Vertex AI 调用"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    stub.ChatVertexAI = ChatVertexAI
    sys.modules['langchain_community.chat_models.vertexai'] = stub
    
    # 延迟注册父模块路径（不强制 import，避免触发惰性加载）
    if 'langchain_community.chat_models' not in sys.modules:
        # 尝试快速注册，失败则不阻塞
        try:
            import langchain_community.chat_models as _lc  # noqa: F811
        except ImportError:
            pass
    return True


RAGAS_AVAILABLE = False
_RAGAS_VERSION = None
_ragas_metrics = {}

try:
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # ── 关键修复：在导入 ragas 前修补 vertexai 依赖 ──
    _patch_vertexai_for_ragas()

    import ragas
    _RAGAS_VERSION = tuple(map(int, ragas.__version__.split('.')))
    print(f"  RAGAS v{ragas.__version__} 已加载")

    from ragas import evaluate as ragas_evaluate

    # 尝试新版导入路径 (RAGAS >= 0.4.0)
    try:
        from ragas.metrics.collections import (
            faithfulness, answer_relevancy, context_precision, context_recall
        )
    except (ImportError, ModuleNotFoundError):
        # 回退到旧版导入路径
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision, context_recall
        )

    _ragas_metrics = {}  # 延迟填充，仅存储 metric 类引用
    # 存储 metric 模块（供延迟实例化使用）
    _ragas_metric_classes = {
        'faithfulness': faithfulness,
        'answer_relevancy': answer_relevancy,
        'context_precision': context_precision,
        'context_recall': context_recall,
    }
    RAGAS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    RAGAS_AVAILABLE = False
    # 区分「未安装」和「依赖缺失」两种情况
    if "ragas" in str(e).lower():
        print(f"  [WARN] RAGAS 未安装，将使用 LLM-as-a-Judge (DeepSeek): {e}")
    else:
        print(f"  [WARN] RAGAS 依赖缺失，将使用 LLM-as-a-Judge (DeepSeek): {e}")
        langchain_v = _get_pkg_version('langchain-community')
        ragas_v = _get_pkg_version('ragas')
        print(f"         langchain-community={langchain_v}, ragas={ragas_v}")
        print(f"         提示: 运行 pip install langchain-community==0.0.21 可修复 RAGAS 支持")
except Exception as e:
    RAGAS_AVAILABLE = False
    print(f"  [WARN] RAGAS 加载异常，回退 LLM-as-a-Judge: {e}")


def load_dataset() -> Dict:
    """加载评估数据集"""
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _detect_port_conflict(port: str):
    """检测端口冲突并输出诊断信息"""
    try:
        import subprocess
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.strip().split()[-1]
                print(f"  [DETECT] 端口 {port} 已被进程 PID={pid} 占用")
                print(f"           请先关闭占用进程或更换端口")
                return True
        else:
            print(f"  [DETECT] 端口 {port} 未被占用，后端服务可能未启动")
            print(f"  [HINT]  启动命令: .venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port {port}")
            return False
    except Exception:
        return None


def _health_check(api_url: str = API_URL, timeout: int = 5) -> bool:
    """检查后端 API 是否可访问
    
    在发起批量评估请求前先做一次健康检查，避免所有请求都失败。
    健康检查超时设置为 5 秒。
    """
    health_url = f"{api_url.rstrip('/')}/health"
    try:
        r = requests.get(health_url, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if 'status' in data and data['status'] == 'ok':
                return True
            print(f"  [WARN] 健康检查返回非正常状态: {data}")
            return True  # 服务有响应即可
        else:
            print(f"  [WARN] 健康检查返回状态码 {r.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  [ERR] 无法连接后端 {api_url}")
        port = api_url.split(':')[-1]
        _detect_port_conflict(port)
        return False
    except requests.exceptions.Timeout:
        print(f"  [ERR] 健康检查超时（{timeout}s）：{health_url}")
        return False
    except Exception as e:
        print(f"  [ERR] 健康检查异常: {e}")
        return False


def _rerank_warmup(api_url: str = API_URL, warmup_timeout: int = 120) -> bool:
    """发送一条测试查询预热 rerank 模型，避免后续请求超时
    
    CrossEncoder 重排序模型在首次使用时需加载到内存（约 10-30 秒）。
    预热请求使用较长的超时时间，确保模型有足够时间加载。
    
    Returns:
        True 表示预热成功（或跳过），False 表示预热失败
    """
    warmup_question = "民法典第188条规定的普通诉讼时效期间是多久？"
    print(f"    [WARMUP] 正在预热 Rerank 模型（超时 {warmup_timeout}s）...")
    
    try:
        resp = requests.post(
            f"{api_url.rstrip('/')}/query",
            json={
                "question": warmup_question,
                "top_k": 5,
                "use_rerank": True,
                "use_keyword_search": True
            },
            timeout=warmup_timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get('found_in_knowledge_base', False):
                print(f"    [WARMUP] Rerank 模型预热成功（{data.get('processing_time_ms', 0):.0f}ms）")
            else:
                print(f"    [WARMUP] 预热完成（未在知识库中找到结果，但 API 正常响应）")
            return True
        else:
            print(f"    [WARMUP] 预热请求返回状态码 {resp.status_code}，跳过预热")
            return True  # 不因预热失败阻塞流程
    except requests.exceptions.Timeout:
        print(f"    [WARMUP] 预热超时（{warmup_timeout}s），模型可能仍在加载中")
        print(f"             后续请求将使用重试机制处理超时")
        return False
    except Exception as e:
        print(f"    [WARMUP] 预热异常: {e}")
        return False


def _query_with_retry(question: str, use_rerank: bool, top_k: int = 5,
                      max_retries: int = 2, base_timeout: Optional[int] = None) -> Optional[Dict]:
    """带指数退避重试的 API 请求（支持 rerank 模式自动延长超时）

    Args:
        question: 问题文本
        use_rerank: 是否使用 Rerank 精排
        top_k: 返回 Top-K 结果
        max_retries: 最大重试次数（幂等重试）
        base_timeout: 首次请求超时时间（秒），默认 None 时根据 use_rerank 自动选择

    超时策略：
      - baseline 模式：首次 60s，重试 120s → 180s
      - rerank  模式：首次 90s，重试 180s → 300s（模型加载 + CrossEncoder 推理）
    
    重试策略（指数退避）：
      - 第1次重试：等待 1 秒（短等待应对临时高负载）
      - 第2次重试：等待 3 秒（稍长等待应对持续高负载）
      - 超过 max_retries 次后直接报告失败

    幂等重试说明：
      - API /query 端点为 GET 风格查询（不修改服务器状态）
      - 超时重试完全安全，不会造成重复写入或状态污染
    """
    if base_timeout is None:
        base_timeout = 90 if use_rerank else 60  # rerank 需要更多时间

    # 退避间隔序列（秒），长度 = max_retries
    backoff_intervals = [1, 3]  # 第1次等1秒，第2次等3秒

    payload = {
        "question": question,
        "top_k": top_k,
        "use_rerank": use_rerank,
        "use_keyword_search": True
    }

    mode_tag = "RERANK" if use_rerank else "BASELINE"

    for attempt in range(max_retries + 1):
        timeout = base_timeout * (attempt + 1)
        try:
            r = requests.post(f"{API_URL}/query", json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            else:
                print(f"    [{mode_tag}] API 返回错误 {r.status_code}: {r.text[:200]}")
                return None  # 非超时错误不重试
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                wait = backoff_intervals[attempt] if attempt < len(backoff_intervals) else 5
                print(f"    [{mode_tag}] [RETRY {attempt+1}/{max_retries}] "
                      f"超时（{timeout}s），{wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"    [{mode_tag}] [ERR] 请求失败（已重试 {max_retries} 次）："
                      f"问题前30字: {question[:30]}..., 超时 {base_timeout}s")
                return None
        except requests.exceptions.ConnectionError as e:
            print(f"    [{mode_tag}] [ERR] 连接失败: {e}")
            # 连接失败后等待 2 秒再返回，让上层有机会切换状态
            time.sleep(2)
            return None
        except requests.exceptions.RequestException as e:
            print(f"    [{mode_tag}] [ERR] 请求异常: {e}")
            return None
    return None


def _deepseek_judge(question: str, answer: str, context: str) -> Dict[str, float]:
    """使用 DeepSeek 作为 Judge LLM，对 answer 进行语义质量打分
    
    返回两个分数（1-5 分制）：
      - answer_relevance: 答案是否回答了用户问题
      - faithfulness: 答案是否忠实于检索到的文档内容（不编造）
    """
    import textwrap
    
    relevance_prompt = textwrap.dedent(f"""\
    你是一个专业的 RAG 评估员。请评估以下「答案」是否回答了用户「问题」。
    
    评分标准（1-5 分）：
    1 分 - 答案完全无关，未回答任何问题
    2 分 - 答案部分相关，但未直接回答核心问题
    3 分 - 答案基本回答了问题，但不够完整或准确
    4 分 - 答案较好地回答了问题，内容准确
    5 分 - 答案完全准确、完整地回答了问题
    
    只需输出 1-5 的数字，不要输出其他内容。
    
    问题：{question}
    答案：{answer}
    """)

    faithfulness_prompt = textwrap.dedent(f"""\
    你是一个专业的 RAG 评估员。请评估以下「答案」是否忠实于提供的「文档内容」，不包含编造的信息。
    
    评分标准（1-5 分）：
    1 分 - 答案完全编造，与文档内容无关
    2 分 - 答案大部分内容没有文档支持
    3 分 - 答案基本忠实，但存在部分推断或编造
    4 分 - 答案几乎完全基于文档，仅少量无关内容
    5 分 - 答案完全基于文档内容，没有任何编造
    
    只需输出 1-5 的数字，不要输出其他内容。
    
    文档内容：{context[:3000]}
    答案：{answer}
    """)
    
    def _call_deepseek(prompt: str) -> Optional[float]:
        try:
            resp = requests.post(
                f"{_DEEPSEEK_BASE_URL}/chat/completions",
                json={
                    "model": _DEEPSEEK_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 5,
                },
                headers={"Authorization": f"Bearer {_DEEPSEEK_API_KEY}"},
                timeout=10,
            )
            if resp.status_code == 200:
                text = resp.json()['choices'][0]['message']['content'].strip()
                score = float(text)
                if 1 <= score <= 5:
                    return score
            return None
        except Exception:
            return None
    
    relevance_score = _call_deepseek(relevance_prompt)
    faithfulness_score = _call_deepseek(faithfulness_prompt)
    
    # 归一化到 [0, 1]
    return {
        "answer_relevance": round((relevance_score or 3.0) / 5.0, 4),
        "faithfulness": round((faithfulness_score or 3.0) / 5.0, 4),
    }


def _compute_self_metrics(results: List[Dict]) -> Dict:
    """LLM-as-a-Judge：RAGAS 不可用时计算替代指标
    
    使用 DeepSeek 作为 Judge LLM 进行语义质量打分：
      - answer_relevance: 答案相关性
      - faithfulness: 忠实度
    同时计算客观指标：
      - knowledge_base_coverage: 知识库覆盖率
      - avg_processing_time: 平均处理时间
    """
    total = len(results)
    if total == 0:
        return {"error": "无评估数据"}

    found_count = sum(1 for r in results if r.get('found_in_knowledge_base', False))
    avg_time = sum(r.get('processing_time_ms', 0) for r in results) / total
    avg_citations = sum(r.get('num_citations', 0) for r in results) / total

    citation_scores = []
    for r in results:
        citation_scores.extend(r.get('citation_scores', []))
    avg_citation_score = sum(citation_scores) / len(citation_scores) if citation_scores else 0
    coverage = found_count / total

    # ── DeepSeek 语义质量打分（逐条评估，花费约 30×2=60 次 API 调用） ──
    relevance_scores = []
    faithfulness_scores = []
    print(f"    [DeepSeek Judge] 正在逐条评估语义质量（共 {total} 条）...")
    for i, r in enumerate(results):
        max_context = r.get('retrieved_contexts', [])
        ctx_str = "\n".join(max_context[:5]) if max_context else "（无检索结果）"
        if not ctx_str.strip():
            ctx_str = "（无检索结果）"
        scores = _deepseek_judge(r['question'], r.get('generated_answer', ''), ctx_str)
        relevance_scores.append(scores['answer_relevance'])
        faithfulness_scores.append(scores['faithfulness'])
        if (i + 1) % 5 == 0:
            print(f"      已评估 {i+1}/{total} 条...")

    def _stats(vals: List[float]) -> Dict:
        import statistics
        return {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    return {
        "metrics_type": "llm_as_a_judge",
        "total_questions": total,
        "found_in_kb": found_count,
        "not_found_in_kb": total - found_count,
        "knowledge_base_coverage": round(coverage, 4),
        "avg_processing_time_ms": round(avg_time, 2),
        "avg_citations_per_query": round(avg_citations, 2),
        "avg_citation_score": round(avg_citation_score, 4),
        "answer_relevance": _stats(relevance_scores),
        "faithfulness": _stats(faithfulness_scores),
        "judge_llm": f"{_DEEPSEEK_MODEL} @ {_DEEPSEEK_BASE_URL}",
        "note": "LLM-as-a-Judge：使用 DeepSeek 对每条 answer 进行语义质量打分。RAGAS 不可用时的主要评估方式。"
    }


def _build_ragas_dataset(results: List[Dict]) -> Optional[Any]:
    """构建 RAGAS 适配的数据集，自动兼容多种 RAGAS 版本

    在 RAGAS 0.4.x 中，evaluate() 接受的 dataset 可以是：
      1. EvaluationDataset（来自 ragas.dataset_schema）— 返回 SingleTurnSample 对象
      2. datasets.Dataset（来自 HuggingFace datasets）— 推荐的兼容方式

    优先使用 EvaluationDataset，若类型校验失败则回退到 HF Dataset 格式。
    RAGAS 0.4.x 的评估指标要求以下列名之一：
      - user_input / question
      - response / answer
      - retrieved_contexts / contexts
      - reference / ground_truth

    Returns:
        EvaluationDataset 或 HF Dataset，或 None（构造失败时）。
    """
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    # ── 类型校验：确保每条 result 包含必要字段 ──
    required_fields = ['question', 'generated_answer', 'ground_truth']
    validated_results = []
    for r in results:
        missing = [f for f in required_fields if not r.get(f)]
        if missing:
            print(f"    [SKIP] 结果缺少字段 {missing}，跳过")
            continue
        # 确保 retrieved_contexts 是 List[str]
        contexts = r.get('retrieved_contexts', [])
        if not isinstance(contexts, list):
            contexts = [str(contexts)] if contexts else []
        r['retrieved_contexts'] = contexts
        validated_results.append(r)

    if not validated_results:
        print("    [WARN] 无有效评估样本")
        return None

    try:
        # 方法1: 使用 EvaluationDataset + SingleTurnSample（RAGAS 原生方式）
        samples = [
            SingleTurnSample(
                user_input=r['question'],
                response=r.get('generated_answer', ''),
                retrieved_contexts=r['retrieved_contexts'],
                reference=r.get('ground_truth', ''),
            )
            for r in validated_results
        ]
        dataset = EvaluationDataset(samples=samples)
        # 验证：确保 dataset[0] 是 SingleTurnSample 而非 dict
        first = dataset[0]
        if isinstance(first, dict):
            raise TypeError(
                f"EvaluationDataset[0] 返回 dict 而非 SingleTurnSample "
                f"(RAGAS 版本兼容问题)"
            )
        print(f"    [OK] 使用 EvaluationDataset（{len(samples)} 条样本）")
        return dataset
    except Exception as e1:
        print(f"    [INFO] EvaluationDataset 方式失败: {e1}")
        print(f"    [INFO] 尝试 HuggingFace Dataset 方式...")

    # 方法2: 使用 HuggingFace datasets.Dataset（更广泛的兼容性）
    try:
        from datasets import Dataset as HFDataset

        data = {
            'user_input': [r['question'] for r in validated_results],
            'response': [r.get('generated_answer', '') for r in validated_results],
            'retrieved_contexts': [r['retrieved_contexts'] for r in validated_results],
            'reference': [r.get('ground_truth', '') for r in validated_results],
        }
        dataset = HFDataset.from_dict(data)
        # 验证：确保列存在
        required_cols = ['user_input', 'response', 'retrieved_contexts', 'reference']
        missing_cols = [c for c in required_cols if c not in dataset.column_names]
        if missing_cols:
            raise ValueError(f"HF Dataset 缺少必要列: {missing_cols}")
        print(f"    [OK] 使用 HuggingFace Dataset（{len(dataset)} 条样本）")
        return dataset
    except ImportError:
        print("    [WARN] HuggingFace datasets 未安装，无法使用 HF Dataset 兜底")
        return None
    except Exception as e2:
        print(f"    [WARN] HF Dataset 方式也失败: {e2}")
        return None


def _run_ragas_evaluation(results: List[Dict]) -> Dict:
    """使用 RAGAS 框架计算忠实度/相关性/精确度/召回率

    兼容 RAGAS 0.4.x 的 API：
      - 使用 SingleTurnSample + EvaluationDataset 或 HuggingFace Dataset
      - 自动检测 RAGAS 版本并选择正确的导入路径
      - 多层兜底：EvaluationDataset → HF Dataset → DeepSeek Judge
    """
    if not RAGAS_AVAILABLE:
        return _compute_self_metrics(results)

    # ── 记录实际样本数 ──
    actual_count = len(results)
    print(f"    [INFO] 待评估样本: {actual_count} 条")

    # ── 构建数据集（含类型校验与多路径兼容） ──
    dataset = _build_ragas_dataset(results)
    if dataset is None:
        print(f"  [WARN] 构造 RAGAS 数据集失败，回退 DeepSeek Judge...")
        return _compute_self_metrics(results)

    # ── 执行 RAGAS 评估 ──
    print(f"  正在计算 RAGAS 指标（数据集 {len(dataset)} 条，{len(results)} 结果）...")

    # ── 延迟实例化 metrics（仅在真正需要 RAGAS 时才加载 embeddings 模型） ──
    if not _ragas_metrics:
        try:
            llm = _init_ragas_llm()
            embeddings = _init_ragas_embeddings()
            if llm is None:
                raise RuntimeError("RAGAS LLM 初始化失败")
            mc = _ragas_metric_classes
            _ragas_metrics['faithfulness'] = mc['faithfulness'].Faithfulness(llm=llm)
            _ragas_metrics['answer_relevancy'] = mc['answer_relevancy'].AnswerRelevancy(
                llm=llm, embeddings=embeddings
            )
            _ragas_metrics['context_precision'] = mc['context_precision'].ContextPrecision(llm=llm)
            _ragas_metrics['context_recall'] = mc['context_recall'].ContextRecall(llm=llm)
        except Exception as init_e:
            print(f"  [WARN] RAGAS metrics 实例化失败: {init_e}")
            print(f"  回退到 LLM-as-a-Judge 方式...")
            return _compute_self_metrics(results)

    metrics_list = [
        _ragas_metrics['faithfulness'],
        _ragas_metrics['answer_relevancy'],
        _ragas_metrics['context_precision'],
        _ragas_metrics['context_recall'],
    ]

    try:
        # raise_exceptions=True 会在评估失败时立即抛出异常，便于调试
        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics_list,
            raise_exceptions=True,
        )
        df = result.to_pandas()

        computed = {}
        metric_keys = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        for key in metric_keys:
            if key in df.columns:
                computed[key] = {
                    "mean": round(float(df[key].mean()), 4),
                    "std": round(float(df[key].std()), 4),
                    "min": round(float(df[key].min()), 4),
                    "max": round(float(df[key].max()), 4),
                }

        metrics = {
            "metrics_type": "ragas",
            **computed,
        }
        print(f"    [OK] RAGAS 评估完成")
        return metrics
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"  [WARN] RAGAS 评估失败: {e}")
        print(f"        版本: RAGAS v{ragas.__version__}, 样本数: {actual_count}")
        print(f"        详细错误:\n{tb}")
        print(f"  回退到 LLM-as-a-Judge 方式...")
        return _compute_self_metrics(results)


def run_eval(mode: str = "both", api_url: str = API_URL):
    """执行评估流水线

    Args:
        mode: baseline | rerank | both
        api_url: 后端 API 地址
    """
    global API_URL
    API_URL = api_url

    print(f"+{'='*60}+")
    print(f"|  LexAI RAGAS 自动化评估流水线")
    print(f"|  API: {API_URL}")
    print(f"|  RAGAS: {'[OK] 已安装' if RAGAS_AVAILABLE else '[NO] 未安装（将用 LLM-as-a-Judge）'}")
    print(f"+{'='*60}+")

    # ── 健康检查：发起批量请求前先确认后端可访问 ──
    print(f"\n  [CHECK] 正在检测后端服务...", end=" ", flush=True)
    if not _health_check(api_url):
        print(f"\n  [ERR] 后端服务不可用，请确认服务已启动: {api_url}")
        print(f"         启动命令: .venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port {api_url.split(':')[-1]}")
        return
    print("OK")

    dataset = load_dataset()
    qa_pairs = dataset['qa_pairs']
    print(f"\n[DATA] 评估数据集: {dataset['metadata']['name']}")
    print(f"   共 {len(qa_pairs)} 条 QA 对，按模式逐条调用后端 API\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 决定评估模式
    modes = []
    if mode in ("baseline", "both"):
        modes.append(("baseline", False))
    if mode in ("rerank", "both"):
        modes.append(("rerank", True))

    for mode_name, use_rerank in modes:
        print(f"\n{'-'*60}")
        print(f"[SEARCH] 模式: {mode_name.upper()} {'(带 Rerank 精排)' if use_rerank else '(基础检索)'}")
        print(f"{'-'*60}")

        # ── Rerank 预热：发送一条测试请求触发模型加载，避免后续请求超时 ──
        if use_rerank:
            print(f"\n  [RERANK] 检测到 Rerank 模式，执行模型预热...")
            _rerank_warmup(api_url)
            print()

        results = []
        errors = 0

        for i, qa in enumerate(qa_pairs, 1):
            question = qa['question']
            print(f"  [{i:02d}/{len(qa_pairs)}] {question[:55]}...", end=" ", flush=True)

            response = _query_with_retry(question, use_rerank=use_rerank)
            if response is None:
                print("[ERR] API 错误")
                errors += 1
                continue

            citations = response.get('citations', [])
            retrieved_contexts = [c.get('content', '') for c in citations]
            citation_scores = [c.get('score', 0.0) for c in citations]

            result_entry = {
                "id": qa['id'],
                "category": qa['category'],
                "question": question,
                "ground_truth": qa['ground_truth'],
                "generated_answer": response.get('answer', ''),
                "found_in_knowledge_base": response.get('found_in_knowledge_base', False),
                "processing_time_ms": response.get('processing_time_ms', 0),
                "retrieved_contexts": retrieved_contexts,
                "citation_scores": citation_scores,
                "num_citations": len(citations)
            }
            results.append(result_entry)

            status = "[OK]" if response.get('found_in_knowledge_base', False) else "[MISS]未找到"
            print(f"{status} ({response.get('processing_time_ms', 0):.0f}ms, {len(citations)} refs)")

        print(f"\n  请求: {len(results)}/{len(qa_pairs)} 成功, {errors} 失败")

        if not results:
            print("  [WARN] 无有效结果，跳过指标计算")
            continue

        # ── 计算评估指标 ──
        print("\n  [CALC] 计算评估指标...")
        if RAGAS_AVAILABLE:
            metrics = _run_ragas_evaluation(results)
        else:
            metrics = _compute_self_metrics(results)

        # ── 按法律类别做交叉分析 ──
        category_stats = {}
        for r in results:
            cat = r['category']
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "found": 0, "avg_time": 0.0, "avg_citations": 0.0}
            category_stats[cat]["total"] += 1
            category_stats[cat]["found"] += 1 if r['found_in_knowledge_base'] else 0
            category_stats[cat]["avg_time"] += r['processing_time_ms']
            category_stats[cat]["avg_citations"] += r['num_citations']
        for cat, stats in category_stats.items():
            n = stats["total"]
            stats["avg_time"] = round(stats["avg_time"] / n, 2)
            stats["avg_citations"] = round(stats["avg_citations"] / n, 2)

        # ── 打包持久化 ──
        output = {
            "mode": mode_name,
            "use_rerank": use_rerank,
            "eval_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset['metadata']['name'],
            "total_questions_dataset": len(qa_pairs),            # 数据集中总样本数
            "total_questions": len(results),                     # 实际处理的样本数
            "successful_queries": len(results),
            "errors": errors,
            "metrics": metrics,
            "category_breakdown": category_stats,
            "per_question_results": results
        }

        output_path = os.path.join(RESULTS_DIR, f"{mode_name}_metrics.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n  [SAVED] 结果已保存: {output_path}")

        _print_metrics_summary(metrics, mode_name)


def _print_metrics_summary(metrics: Dict, mode_name: str):
    """打印指标摘要到控制台"""
    print(f"\n  [CHART] {mode_name.upper()} 评估摘要:")
    print(f"  {'-'*42}")

    if metrics.get('metrics_type') == 'ragas':
        for key, label in [
            ("faithfulness", "Faithfulness (忠实度)"),
            ("answer_relevancy", "Answer Relevancy (答案相关性)"),
            ("context_precision", "Context Precision (上下文精确度)"),
            ("context_recall", "Context Recall (上下文召回率)"),
        ]:
            m = metrics.get(key, {})
            if m:
                print(f"    {label}")
                print(f"      Mean: {m['mean']:.4f}  Std: {m['std']:.4f}")
                print(f"      Min:  {m['min']:.4f}  Max: {m['max']:.4f}")
    else:
        print(f"    Knowledge Base Coverage: {metrics.get('knowledge_base_coverage', 'N/A')}")
        print(f"    Avg Processing Time:     {metrics.get('avg_processing_time_ms', 'N/A')} ms")
        print(f"    Avg Citations / Query:   {metrics.get('avg_citations_per_query', 'N/A')}")
        print(f"    Avg Citation Score:      {metrics.get('avg_citation_score', 'N/A')}")
        # ── DeepSeek Judge 语义质量指标 ──
        for key, label in [
            ("answer_relevance", "Answer Relevance (DeepSeek Judge)"),
            ("faithfulness", "Faithfulness (DeepSeek Judge)"),
        ]:
            m = metrics.get(key, {})
            if m:
                print(f"    {label}")
                print(f"      Mean: {m['mean']:.4f}  Std: {m['std']:.4f}")
                print(f"      Min:  {m['min']:.4f}  Max: {m['max']:.4f}")
        judge_llm = metrics.get('judge_llm', '')
        if judge_llm:
            print(f"    Judge LLM: {judge_llm}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LexAI RAGAS 自动化评估流水线")
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["baseline", "rerank", "both"],
        help="评估模式: baseline (无 Rerank) | rerank (带 Rerank) | both (两种都跑)"
    )
    parser.add_argument(
        "--api-url", type=str, default=API_URL,
        help="后端 API 地址（默认从 backend.config 读取）"
    )
    args = parser.parse_args()
    run_eval(mode=args.mode, api_url=args.api_url)
