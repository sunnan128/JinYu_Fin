# ── 向量检索服务 ──
# 决策记录：
# - 6 层降级策略加载嵌入模型：本地缓存 → HF 镜像 → 轻量模型
# - BM25 关键词索引持久化到磁盘 pickle，不在 RAM 中常驻
# - 混合检索策略演进：加权求和(0.7+0.3) → RRF 融合 → RRF + 层级关联评分
# - Bug 修复: add_documents 中 metadatas 索引原为 len(all_docs)-len(chunks)，
#   上传第二个文件时越界。已改为 enumerate(chunks) 正确索引。
# - 层级关联评分（2026-07）：
#   问题：条款块正文不含法律名称（如"第三十五条…"块中无"民法典"），
#         BM25/语义均无法匹配，top-20 候选池不含目标文档。
#   方案：_parse_hierarchical_query → 元数据补充检索 → _apply_hierarchical_scoring
#         + 条款号归一化双向转换（阿拉伯↔中文数字）。
#   关键设计：候选池参数 rerank_candidates=20 保持不变。

import os
import re
import pickle
import uuid
from typing import List, Dict, Any, Optional, Tuple
from backend.config import settings
from backend.utils.document_parser import DocumentChunk

# ── LangSmith 全链路追踪（条件化集成） ──
if settings.LANGSMITH_TRACING:
    # 将配置注入环境变量，langsmith SDK 从 os.environ 读取
    os.environ.setdefault("LANGSMITH_API_KEY", settings.LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    from langsmith import traceable
else:
    # 关闭时使用无操作装饰器，零额外开销
    def traceable(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda x: x

# 项目根目录（确保路径解析正确）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hugging Face 镜像配置（国内用户无法直连 HF）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = os.path.join(_PROJECT_ROOT, 'backend', 'data', 'model_cache')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(_PROJECT_ROOT, 'backend', 'data', 'model_cache')

MODEL_CACHE_DIR = os.path.join(_PROJECT_ROOT, 'backend', 'data', 'model_cache')
# BM25 索引持久化路径（存在磁盘，按需加载到内存）
BM25_PERSIST_PATH = os.path.join(_PROJECT_ROOT, 'backend', 'data', 'bm25_index.pkl')

# Import optional dependencies with error handling
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("Warning: sentence-transformers not installed. Embedding features will not work.")

try:
    import chromadb
except ImportError:
    chromadb = None
    print("Warning: chromadb not installed. Vector storage will not work.")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    print("Warning: rank-bm25 not installed. Keyword search will not work.")

try:
    import jieba
except ImportError:
    jieba = None
    print("Warning: jieba not installed. Chinese text segmentation will not work.")

try:
    import numpy as np
except ImportError:
    np = None
    print("Warning: numpy not installed.")

def _download_model_from_modelscope(model_name: str, cache_dir: str) -> str:
    """Download model from ModelScope to local cache directory"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        os.makedirs(cache_dir, exist_ok=True)
        model_dir = snapshot_download(model_name, cache_dir=cache_dir)
        return model_dir
    except Exception as e:
        print(f"Warning: Could not download from ModelScope: {e}")
        return None

def _check_hf_cache(model_name: str) -> Optional[str]:
    """Check if model exists in HuggingFace hub cache structure
    
    HF cache uses format: models--org--model_name/snapshots/<hash>/
    Returns the snapshot path if found, None otherwise.
    """
    hf_dir_name = "models--" + model_name.replace("/", "--")
    hf_cache_path = os.path.join(MODEL_CACHE_DIR, hf_dir_name)
    
    # Read refs/main to get current snapshot hash
    refs_path = os.path.join(hf_cache_path, "refs", "main")
    if not os.path.isfile(refs_path):
        return None
    
    try:
        with open(refs_path, 'r') as f:
            snapshot_hash = f.read().strip()
    except Exception:
        return None
    
    snapshot_path = os.path.join(hf_cache_path, "snapshots", snapshot_hash)
    if os.path.isdir(snapshot_path) and os.path.exists(os.path.join(snapshot_path, "config.json")):
        return snapshot_path
    
    return None

def _download_default_embedding_model():
    """Check if the default embedding model is available in local cache"""
    actual_model_name = getattr(settings, 'EMBEDDING_MODEL', 'BAAI/bge-large-zh-v1.5')
    
    # Check HuggingFace hub cache structure (preferred)
    snapshot_path = _check_hf_cache(actual_model_name)
    if snapshot_path:
        print(f"Found model in HF cache: {snapshot_path}")
        return snapshot_path
    
    # Fallback: check flat directory structure
    flat_path = os.path.join(MODEL_CACHE_DIR, actual_model_name.replace("/", "_"))
    if os.path.isdir(flat_path) and os.path.exists(os.path.join(flat_path, "config.json")):
        print(f"Found model in flat cache: {flat_path}")
        return flat_path
    
    print(f"Model not found in local cache (offline mode)")
    return None

# ── 层级关联检索：条款号归一化辅助 ──
# 决策记录：
# - 动机：用户输入"第35条"（阿拉伯数字），文档写"第三十五条"（中文数字），
#   BM25 分词后 token 完全不匹配，分数为 0。
# - 方案：对"第N条/款/章/节"格式的字符串，实现阿拉伯↔中文数字双向转换。
# - 与 qa_service.py 中已有归一化的关系：qa_service 是查询侧单向转换，
#   此处是评分侧双向检查，确保无论用户输入哪种格式都能匹配文档内容。
# - 边界情况：支持"十""百"等位数词，"一十"自动修正为"十"。

_ARABIC_DIGITS = '0123456789'
_CHINESE_NUM_SIMPLE = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']

def _number_to_chinese(num_str: str) -> str:
    """将阿拉伯数字串转为中文数字（支持条款号场景）
    
    示例："" → "",  "1" → "一",  "35" → "三十五",  "100" → "一百"
    """
    if not num_str:
        return ''
    n = int(num_str)
    if n == 0:
        return '零'
    units = ['', '十', '百', '千']
    result = ''
    digits = [int(d) for d in str(n)]
    length = len(digits)
    for i, d in enumerate(digits):
        pos = length - 1 - i
        if d == 0:
            # 零只出现在非空结果且前一位非零时
            if result and not result.endswith('零'):
                result += '零'
        else:
            result += _CHINESE_NUM_SIMPLE[d]
            if pos > 0:
                result += units[pos]
    # 去除末尾多余的零
    result = result.rstrip('零')
    # 处理"一十"开头的情况（如 10 → "十" 而非 "一十"）
    if result.startswith('一十'):
        result = result[1:]
    return result

def _chinese_to_number(chinese: str) -> Optional[str]:
    """将中文数字转为阿拉伯数字字符串（支持条款号场景）
    
    示例："" → None, "一" → "1",  "三十五" → "35",  "一百" → "100"
    """
    if not chinese:
        return None
    _cn_map = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
               '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    _unit_map = {'十': 10, '百': 100, '千': 1000}
    try:
        total = 0
        current = 0
        for ch in chinese:
            if ch in _cn_map:
                current = _cn_map[ch]
            elif ch in _unit_map:
                if current == 0:
                    current = 1
                total += current * _unit_map[ch]
                current = 0
            else:
                return None
        total += current
        return str(total)
    except:
        return None

def _normalize_article_variant(article_str: str) -> Optional[str]:
    """生成条款号的另一种书写形式（阿拉伯数字 ↔ 中文数字）
    
    示例：
      "第35条"    → "第三十五条"
      "第三十五条" → "第35条"
      "第1条"     → "第一条"
      "第一条"    → "第1条"
      "第1章"     → "第一章"
      "第100条"   → "第一百条"
    """
    match = re.match(r'^(第)(\d+)([条款章节])$', article_str)
    if match:
        prefix, num_str, suffix = match.groups()
        chinese_num = _number_to_chinese(num_str)
        return f'{prefix}{chinese_num}{suffix}'
    
    chinese_digits = set('零一二三四五六七八九十百千')
    match = re.match(r'^(第)([零一二三四五六七八九十百千]+)([条款章节])$', article_str)
    if match:
        prefix, num_str, suffix = match.groups()
        arabic = _chinese_to_number(num_str)
        if arabic is not None:
            return f'{prefix}{arabic}{suffix}'
    
    return None

class VectorStoreService:
    def __init__(self):
        # Check required dependencies
        if chromadb is None:
            raise ImportError("chromadb is not installed. Please install it with 'pip install chromadb'.")
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Please install it with 'pip install sentence-transformers'.")
        if jieba is None:
            raise ImportError("jieba is not installed. Please install it with 'pip install jieba'.")
        if BM25Okapi is None:
            raise ImportError("rank-bm25 is not installed. Please install it with 'pip install rank-bm25'.")
        
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name="financial_documents",
            metadata={"description": "金融文档知识库"}
        )
        
        # Load embedding model
        self.embedding_model = self._load_embedding_model()
        
        self.bm25_index = None
        self.all_documents = []
        self._load_bm25_from_disk()
    
    def _load_embedding_model(self):
        """Load embedding model with multiple fallback strategies
        
        Uses HF_HUB_OFFLINE=1 to prevent downloads when model is cached locally.
        """
        # Strategy 1: Try to load from local HF cache snapshot path
        local_model_path = _download_default_embedding_model()
        if local_model_path:
            try:
                print(f"Loading model from local cache: {local_model_path}...")
                model = SentenceTransformer(local_model_path)
                print("Model loaded successfully from local cache!")
                return model
            except Exception as e:
                print(f"Could not load from local path: {e}")
        
        # Strategy 2: Try specified model by name (uses HF cache)
        print(f"Trying to load model: {settings.EMBEDDING_MODEL}...")
        try:
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
        
        # Strategy 3: Fallback to lightweight model
        print("Trying lightweight fallback model...")
        try:
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            print("Lightweight model loaded!")
            return model
        except Exception as e:
            print(f"All model loading strategies failed: {e}")
            raise RuntimeError(
                "No embedding model could be loaded. "
                "Please run 'python download_model.py' or copy model_cache from another project."
            )
    
    def _load_bm25_from_disk(self):
        """从磁盘 pickle 加载 BM25 索引（不在 RAM 中常驻重建）"""
        if os.path.exists(BM25_PERSIST_PATH):
            try:
                with open(BM25_PERSIST_PATH, 'rb') as f:
                    data = pickle.load(f)
                self.all_documents = data['documents']
                if self.all_documents:
                    tokenized_docs = [jieba.lcut(doc['content']) for doc in self.all_documents]
                    self.bm25_index = BM25Okapi(tokenized_docs)
                print(f"BM25 索引已从磁盘加载，共 {len(self.all_documents)} 个片段")
            except Exception as e:
                print(f"BM25 磁盘加载失败，将从 ChromaDB 重建: {e}")
                self._rebuild_bm25_from_db()
        else:
            print("未找到 BM25 持久化文件，将从 ChromaDB 重建")
            self._rebuild_bm25_from_db()
    
    def _rebuild_bm25_from_db(self):
        """从 ChromaDB 重建 BM25 索引（兜底方案）"""
        results = self.collection.get()
        self.all_documents = []
        
        if results and results['documents']:
            for i, doc in enumerate(results['documents']):
                self.all_documents.append({
                    'id': results['ids'][i],
                    'content': doc,
                    'metadata': results['metadatas'][i] if results['metadatas'] else {}
                })
            
            if self.all_documents:
                tokenized_docs = [jieba.lcut(doc['content']) for doc in self.all_documents]
                self.bm25_index = BM25Okapi(tokenized_docs)
        
        self._save_bm25_to_disk()
    
    def _save_bm25_to_disk(self):
        """将 BM25 文档数据持久化到磁盘 pickle"""
        try:
            os.makedirs(os.path.dirname(BM25_PERSIST_PATH), exist_ok=True)
            with open(BM25_PERSIST_PATH, 'wb') as f:
                pickle.dump({'documents': self.all_documents}, f)
        except Exception as e:
            print(f"BM25 持久化失败（不影响运行）: {e}")
    
    def add_documents(self, chunks: List[DocumentChunk], document_id: str, filename: str) -> int:
        if not chunks:
            return 0
        
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            meta = {
                "document_id": document_id,
                "filename": filename,
                "page_number": chunk.page_number or 0,
                "paragraph_number": chunk.paragraph_number or 0
            }
            meta.update(chunk.metadata)
            metadatas.append(meta)
        
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True).tolist()
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        for i, chunk in enumerate(chunks):
            self.all_documents.append({
                'id': chunk.id,
                'content': chunk.content,
                'metadata': metadatas[i]
            })
        
        if self.all_documents:
            tokenized_docs = [jieba.lcut(doc['content']) for doc in self.all_documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
        
        self._save_bm25_to_disk()
        
        return len(chunks)
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        formatted_results = []
        if results and results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': results['distances'][0][i] if results['distances'] else 0.0
                })
        
        return formatted_results
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25_index or not self.all_documents:
            return []
        
        tokenized_query = jieba.lcut(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.all_documents[idx]
                results.append({
                    'id': doc['id'],
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': float(scores[idx])
                })
        
        return results
    
    # ── 层级关联查询解析 ──
    # 决策记录：
    # - 自动识别"法律名称+条款序号"类查询（如"民法典第35条"），
    #   提取主体关键词"民法典"和子关键词"第35条"。
    # - 仅适用于中文法律检索场景，正则匹配 "第X条/款/章/节" 格式。
    # - 未匹配到时返回 (None, None)，hybrid_search 回退到旧版评分逻辑。
    # - 多字法律名称也支持（如"中华人民共和国消费者权益保护法第35条"）。

    @staticmethod
    def _parse_hierarchical_query(query: str) -> Tuple[Optional[str], Optional[str]]:
        """解析层级关联查询，提取「主体限定关键词」与「子匹配关键词」
        
        自动识别法律法规检索场景中"法律名称 + 条款序号"的结构化查询：
          "民法典第35条"   → subject="民法典",  sub_key="第35条"
          "民法典第三十五条" → subject="民法典",  sub_key="第三十五条"
          "第35条"         → subject=None,    sub_key="第35条"
          "第三十五条"     → subject=None,    sub_key="第三十五条"
        
        Returns:
            (subject_key, sub_key) — 未检测到层级结构时返回 (None, None)
        """
        article_match = re.search(r'(第[零一二三四五六七八九十百千\d]+[条款章节])', query)
        if not article_match:
            return (None, None)
        
        sub_key = article_match.group(1)
        before = query[:article_match.start()].strip()
        
        if not before:
            return (None, sub_key)
        
        return (before, sub_key)

    # ── 层级化相关度评分 ──
    # 决策记录：
    # - 主体判定依据：先检查元数据（文件名），再检查文档正文。
    #   因为条款块正文不含法律名称（如"第三十五条…"块中无"民法典"），
    #   但文件名含"中华人民共和国民法典_20200528.docx"。
    # - 子关键词检查自动归一化（阿拉伯↔中文数字双向匹配）。
    # - 三个梯次的加分值是经验值，基于 RRF 分数范围（≈0.017-0.095）确定：
    #   0.03 确保 TIER-1 能超越大多数 RRF 结果，
    #   0.01 使 TIER-2 略高于纯 RRF 中位水平，但低于 TIER-1。
    # - 不修改候选池参数（rerank_candidates=20），仅在评分侧做调整。

    @staticmethod
    def _apply_hierarchical_scoring(
        scored_items: List[Dict],
        subject_key: Optional[str],
        sub_key: Optional[str],
    ) -> None:
        """应用层级化相关度评分

        实现"先限定主体范围，再匹配子关键词"的关联检索逻辑：
        
        主体判定依据：先检查文档内容的元数据（文件名），再检查文档正文。
        这是因为法律文档的条款块内容通常不包含该法律的完整名称
        （例如"第三十五条　监护人应按照…"块中不含"民法典"字样），
        但文档文件名包含法律名称（如"中华人民共和国民法典_20200528.docx"）。
        
        子关键词自动归一化：同时检查阿拉伯数字和中文数字两种形式
        （例如 sub_key="第35条" 也会匹配 "第三十五条"），
        解决查询词与文档内容数字书写形式不一致的问题。
        
        **评分梯次**（从高到低）：
          梯次①  主体 + 子关键词  →  base + 0.03  (高相关，精准命中)
          梯次②  仅主体关键词    →  base + 0.01  (中等相关，在同一法律的文档内)
          梯次③  仅子关键词/均无 →  base          (低相关，其他法律的同名条款)
        
        **场景示例**：查询"民法典第35条"
          - 文件名含"民法典" + 内容含"第三十五条"/"第35条" → 梯次① ✓ 精准命中
          - 文件名含"民法典"但内容无"第三十五条"/"第35条" → 梯次② 同一法律的其它条款
          - 文件名不含"民法典"但内容含"第三十五条"/"第35条" → 梯次③ 其他法律同名条款
        """
        # 任一关键词为 None 时不执行层级评分
        if subject_key is None or sub_key is None:
            return
        
        FULL_MATCH_BOOST = 0.03    # 主体+子关键词，超越任何 RRF 单列得分
        SUBJECT_ONLY_BOOST = 0.01  # 仅主体关键词，落后于全命中文档但领先于其他
        
        # 子关键词自动归一化：生成阿拉伯数字和中文数字两个变体
        sub_variants = [sub_key]
        alt_sub = _normalize_article_variant(sub_key)
        if alt_sub and alt_sub != sub_key:
            sub_variants.append(alt_sub)
        
        for item in scored_items:
            content = item['content']
            # 主体判定：从元数据（文件名）和文档内容两方面检查
            metadata = item.get('metadata', {})
            filename = metadata.get('filename', '') if isinstance(metadata, dict) else ''
            
            has_subject = (subject_key in content) or (subject_key in filename)
            has_sub = any(v in content for v in sub_variants)
            
            if has_subject and has_sub:
                item['score'] += FULL_MATCH_BOOST
            elif has_subject:
                item['score'] += SUBJECT_ONLY_BOOST
            # 仅子关键词或都不命中：不额外加分

    def _search_by_subject_metadata(self, subject_key: str) -> List[Dict[str, Any]]:
        """通过元数据（文件名）检索归属于某法律主体的所有文档块
        
        法律文档的条款块内容通常不包含该法律的完整名称
        （如"第三十五条　监护人应按照…"块中不含"民法典"字样），
        但文件名中包含法律名称（如"..._中华人民共和国民法典_20200528.docx"）。
        此方法通过文件名匹配来补充主体相关的文档块到候选池中，
        确保层级检索不会遗漏那些正文不含主体名的条款块。
        
        Returns:
            匹配的文档块列表（每项含 id/content/metadata，score 初始化为 0）
        """
        if not self.all_documents:
            return []
        results = []
        for doc in self.all_documents:
            meta = doc.get('metadata', {})
            filename = meta.get('filename', '') if isinstance(meta, dict) else ''
            if subject_key in filename:
                results.append({
                    'id': doc['id'],
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': 0.0  # 基础分，后续通过层级评分调整
                })
        return results

    @traceable(run_type="retriever")
    def hybrid_search(self, query: str, top_k: int = 5, 
                     semantic_weight: float = 0.7, 
                     keyword_weight: float = 0.3,
                     rerank_candidates: int = 0,
                     use_keyword: bool = True) -> List[Dict[str, Any]]:
        # ── 步骤1: 解析层级关联查询 ──
        subject_key, sub_key = self._parse_hierarchical_query(query)
        
        # ── 步骤2: RRF 混合检索 ──
        # 当启用 rerank 时，对内部语义/关键词检索取更多候选
        internal_k = max(top_k * 2, rerank_candidates) if rerank_candidates > 0 else top_k * 2
        semantic_results = self.semantic_search(query, internal_k)
        keyword_results = self.keyword_search(query, internal_k) if use_keyword else []
        
        # Reciprocal Rank Fusion (RRF)：基于排序位置融合，不依赖分数量级归一化
        # RRF(d) = 1/(k + rank_sem(d)) + 1/(k + rank_kw(d))
        RRF_K = 20
        
        sem_rank = {r['id']: i + 1 for i, r in enumerate(semantic_results)}
        kw_rank = {r['id']: i + 1 for i, r in enumerate(keyword_results)}
        
        all_ids = set(sem_rank.keys()) | set(kw_rank.keys())
        
        scored = []
        for doc_id in all_ids:
            rank_sem = sem_rank.get(doc_id)
            rank_kw = kw_rank.get(doc_id)
            
            rrf_score = 0.0
            if rank_sem is not None:
                rrf_score += 1.0 / (RRF_K + rank_sem)
            if rank_kw is not None:
                rrf_score += 1.0 / (RRF_K + rank_kw)
            
            if rank_sem is not None:
                src = semantic_results[rank_sem - 1]
            else:
                src = keyword_results[rank_kw - 1]
            
            scored.append({
                'id': doc_id,
                'content': src['content'],
                'metadata': src['metadata'],
                'score': rrf_score
            })
        
        # ── 步骤3: 主体元数据补充检索 ──
        # 决策记录：
        # - 动机：条款块正文不含法律名称时 BM25/语义均无法命中，
        #   即使 internal_k=40 也无法将目标文档纳入候选池。
        # - 方案：通过文件名主动查找该法律所有文档块加入候选池，
        #   基础分 0.02（≈RRF排名第30的单路分值），配合层级评分后合理排序。
        # - 不修改 rerank_candidates=20，只在候选池内调整排序。
        # - 如果目标文档已通过 RRF 进入候选池，不会重复添加（去重）。
        if subject_key:
            subject_docs = self._search_by_subject_metadata(subject_key)
            existing_ids = {item['id'] for item in scored}
            for sd in subject_docs:
                if sd['id'] not in existing_ids:
                    sd['score'] = 0.02  # 中等基础分，配合层级评分后合理排序
                    scored.append(sd)
        
        # ── 步骤4: 层级或关键词加分 ──
        if subject_key and sub_key:
            # 场景①：检测到"主体+子关键词"的层级结构
            # 使用层级化评分，确保同时命中主体和子关键词的文档排在最前
            self._apply_hierarchical_scoring(scored, subject_key, sub_key)
        elif sub_key:
            # 场景②：仅子关键词，无主体限定
            # 兼容处理：对包含该条款号的文档给予基础加分
            article_boost = 1.0 / RRF_K  # 0.05
            for item in scored:
                content = item['content']
                alt_sub = _normalize_article_variant(sub_key)
                sub_variants = [sub_key]
                if alt_sub and alt_sub != sub_key:
                    sub_variants.append(alt_sub)
                if any(v in content for v in sub_variants):
                    item['score'] += article_boost
        
        # ── 步骤5: 最终排序与截断 ──
        final_k = rerank_candidates if rerank_candidates > 0 else top_k
        scored.sort(key=lambda x: x['score'], reverse=True)
        
        return scored[:final_k]
    
    def delete_document(self, document_id: str) -> int:
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        if results and results['ids']:
            self.collection.delete(ids=results['ids'])
            self._rebuild_bm25_from_db()
            return len(results['ids'])
        
        return 0
    
    def get_document_chunks(self, document_id: str, page: int = 1, page_size: int = 50) -> Dict:
        """获取文档的所有检索片段（分页），用于文档预览"""
        results = self.collection.get(
            where={"document_id": document_id}
        )
        if not results or not results['ids']:
            return {"total": 0, "chunks": []}
        
        # 按页码和段落号排序
        chunks = []
        for i in range(len(results['ids'])):
            meta = results['metadatas'][i] if results['metadatas'] else {}
            chunks.append({
                "id": results['ids'][i],
                "content": results['documents'][i] if results['documents'] else "",
                "page_number": meta.get('page_number', 0),
                "paragraph_number": meta.get('paragraph_number', 0),
            })
        
        chunks.sort(key=lambda c: (c['page_number'], c['paragraph_number']))
        
        total = len(chunks)
        start = (page - 1) * page_size
        end = start + page_size
        page_chunks = chunks[start:end]
        
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size,
            "chunks": page_chunks
        }
    
    def get_document_count(self) -> int:
        return self.collection.count()
    
    def clear_all(self):
        self.client.delete_collection("legal_documents")
        self.collection = self.client.get_or_create_collection(
            name="legal_documents",
            metadata={"description": "法律文档知识库"}
        )
        self.all_documents = []
        self.bm25_index = None
        if os.path.exists(BM25_PERSIST_PATH):
            os.remove(BM25_PERSIST_PATH)
