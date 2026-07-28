# 金语AI 金融知识问答系统

> **RAG 增强 · 知识图谱 · 溯源可信 · 安全合规**  
> 基于检索增强生成（RAG）技术的金融与法律领域智能问答系统。

---

## 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [环境依赖](#环境依赖)
- [快速部署](#快速部署)
  - [方式一：一键启动（推荐）](#方式一一键启动推荐)
  - [方式二：分步启动](#方式二分步启动)
  - [方式三：Docker 部署](#方式三docker-部署)
- [常用命令](#常用命令)
- [项目结构](#项目结构)
- [文档切分策略（边界感知）](#文档切分策略边界感知)
- [三层幻觉抑制与时效风险提醒](#三层幻觉抑制与时效风险提醒)
- [技术栈](#技术栈)
- [API 文档](#api-文档)
- [配置说明](#配置说明)
- [模型下载](#模型下载)
- [贡献者规范](#贡献者规范)
- [许可证](#许可证)
- [问题反馈](#问题反馈)

---

## 项目简介

金语AI 是一套专注于金融与法律领域的 **RAG（检索增强生成）知识问答系统**。系统基于混合检索技术（语义检索 + 关键词检索 + 知识图谱），从上传的专业文档中精确定位相关内容，再通过大语言模型生成带引用来源的可靠答案。**"未检索到即明确告知，绝不编造"** 是系统的核心设计原则。

### 适用场景

- 金融机构内部法规知识库查询
- 律师事务所法律条文检索与引用
- 金融研究员文档数据快速问答
- 合规部门监管政策解读

---

## 核心功能

| 功能 | 描述 |
|------|------|
| **多格式文档上传** | 支持 PDF、Word（.docx）、Markdown（.md）文档上传与解析 |
| **智能问答** | 基于 RAG 技术的自然语言问答，支持三种检索模式 |
| **混合检索** | 语义检索 + BM25 关键词 + 知识图谱 三路召回，RRF 融合 |
| **重排序精排** | 使用 CrossEncoder 对候选结果重排序，提升相关性 |
| **层级关联评分** | 法律名称与条款号的层级匹配加权（如"反洗钱法第58条"） |
| **知识图谱检索** | 金融实体关系图谱查询（Neo4j 可选，支持 Mock 降级） |
| **精准溯源** | 每个答案附带引用来源（文档名/章节/条款/页码），可点击查看原文 |
| **异步上传** | 大文档异步处理，前端实时进度反馈 |
| **拒绝幻觉** | 未在知识库中检索到相关内容时，明确告知用户 |
| **文档预览** | 支持按段落号跳转的 HTML 文档片段预览页 |
| **合规保障** | 常驻免责声明，明确系统非专业金融建议 |
| **三层幻觉抑制** | L1 检索端权威度/时效过滤、L2 Prompt 强约束、L3 生成后校验（比对已废止条款对照表） |
| **时效风险提醒** | 命中已废止/失效条款时前端展示 ⚠️ 提醒（含发行/废止日期、明文废止依据、官方原文链接） |

---

## 环境依赖

### 系统要求

- **操作系统**：Windows 10+ / macOS / Linux
- **Python**：3.10 ~ 3.11
- **磁盘空间**：至少 4GB（含嵌入模型缓存 ~2.5GB）
- **内存**：至少 8GB（推荐 16GB）
- **网络**：首次运行需下载嵌入模型与重排序模型；问答功能需连接 LLM API

### Python 依赖

核心依赖（详见 `requirements.txt`）：

```
fastapi==0.109.0       # 后端框架
uvicorn[standard]==0.27.0   # ASGI 服务器
streamlit==1.29.0      # 前端框架
chromadb==0.4.22       # 向量数据库
sentence-transformers==2.3.1  # 嵌入模型
rank-bm25==0.2.2       # BM25 关键词检索
pymupdf==1.23.8        # PDF 解析
python-docx==1.1.0     # Word 解析
jieba                  # 中文分词
openai==1.10.0         # LLM API 调用
numpy==1.26.3          # 数值计算
neo4j                  # 图数据库（可选）
langsmith>=0.1.0       # 追踪（可选）
```

---

## 快速部署

### 方式一：一键启动（推荐）

确保已安装 Python 3.10+ 并配置好 `.env` 文件。

```bash
# 1. 克隆项目
git clone <项目地址>
cd <项目目录>

# 2. 创建虚拟环境（推荐）
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 配置环境变量
copy .env.example .env     # Windows
# cp .env.example .env     # macOS/Linux
# 编辑 .env 填入 LLM API Key 等信息

# 5. 下载嵌入模型（首次运行必须）
python download_model.py

# 6. 一键启动
start_jinyu.bat            # Windows
```

启动后访问：
- **前端**：http://localhost:8506
- **后端 API**：http://localhost:8006
- **API 文档**：http://localhost:8006/docs

### 方式二：分步启动

```bash
# 终端 1：启动后端（默认端口 8006）
uvicorn backend.main:app --host 0.0.0.0 --port 8006 --reload

# 终端 2：启动前端（默认端口 8506）
streamlit run frontend/app_jinyu.py --server.port 8506 --server.headless true
```

### 方式三：Docker 部署

```bash
# 构建镜像并启动
docker-compose up -d --build

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

> **注意**：Docker 部署时需确保 `.env` 文件已配置，且 `backend/data` 目录有写入权限。

---

## 常用命令

### 依赖管理

```bash
# 安装依赖（清华镜像源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖（自动脚本）
python install_dependencies.py
```

### 模型下载

```bash
# 下载嵌入模型（首次运行必须）
python download_model.py
```

### 启动服务

```bash
# 一键启动（Windows）
start_jinyu.bat

# 单独启动后端
uvicorn backend.main:app --host 0.0.0.0 --port 8006 --reload

# 单独启动前端
streamlit run frontend/app_jinyu.py --server.port 8506 --server.headless true
```

### 重启后端

```bash
# 自动检测端口占用并重启
python restart_backend.py
```

### 运行评估

```bash
# 运行 RAGAS 自动化评估
python -m eval.run_eval --mode both

# 运行评估对比分析
python eval/comparison.py
```

### 单元测试

```bash
# 实体链接器单元测试
python backend/utils/entity_linker.py

# 混合检索器单元测试
python backend/services/hybrid_retriever.py

# 三层幻觉抑制守卫 / 接线集成 / 前端提醒条 全套测试（共 24 项）
python -m pytest tests -v
```

> 测试无需真实模型与 API：底层向量库与 LLM 以 Fake 对象替换，覆盖上传/解析/检索/L1 拦截/L3 拦截-打标-重答/提醒条渲染等全链路。

---

## 项目结构

```
project_root/
├── backend/                      # 后端服务
│   ├── config.py                 # 配置管理
│   ├── main.py                   # FastAPI 应用入口
│   ├── models/
│   │   └── schemas.py            # 数据模型定义
│   ├── services/
│   │   ├── qa_service.py         # 查询路由中枢
│   │   ├── vector_store.py       # 向量存储 + BM25 + RRF
│   │   ├── llm_service.py        # 重排序 + LLM 生成
│   │   ├── hybrid_retriever.py   # 三路混合检索器
│   │   └── graph_service.py      # 图谱检索服务
│   ├── utils/
│   │   ├── document_parser.py    # 文档解析与切分（边界感知 + front-matter）
│   │   ├── entity_linker.py      # 金融实体链接
│   │   ├── hallucination_guard.py # 三层幻觉抑制守卫（L1/L2/L3）
│   │   └── superseded.json       # 已废止/失效条款对照表
│   └── data/                     # 数据目录
│       ├── chroma/               # ChromaDB 持久化
│       ├── model_cache/          # 模型缓存
│       └── bm25_index.pkl        # BM25 索引
├── frontend/                     # 前端应用
│   ├── app.py                    # 默认前端
│   ├── app_jinyu.py              # 金语品牌前端（主推，接 ⚠️ 时效风险提醒）
│   ├── app_xinglin.py            # 杏林品牌前端
│   └── guard_banner.py           # 时效风险提醒条 HTML 渲染
├── sample_test_docs/             # 端到端测试样例文档
├── eval/                         # 评估套件
│   ├── dataset.json              # 20条评估数据集
│   ├── run_eval.py               # RAGAS 自动化评估
│   ├── comparison.py             # 对比分析
│   └── results/                  # 评估结果输出
├── .env.example                  # 环境变量模板
├── .env                          # 环境变量（请勿上传）
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── download_model.py             # 模型下载
├── install_dependencies.py       # 依赖安装
├── restart_backend.py            # 后端重启
└── start_jinyu.bat               # 一键启动
```

---

## 文档切分策略（边界感知）

系统以「条」为最小 chunk 单位，采用**边界感知（boundary-aware）切分**，确保一条完整法条（如《反洗钱法》第五十五条、第五十六条）始终成块，不被误切：

- **边界判断**：识别「第X条」为新法条起点时，要求前导字符为句末标点（。；！？）、换行、全角空格（`\u3000`）或右括号；「本法第五十三条」「依照本法第五十二条至第五十四条规定」等**行内引用**不会触发新块。
- **长条细分**：单条超过阈值（约 700 字）时按「款」（中文分号 `；`）细分，每块带条号前缀，溯源不丢失。
- **兜底策略**：不含「第X条」的文档（人大决定 / 公告）整篇作为一块，不丢内容。
- **元数据剥离**：剔除 chunk 正文内的 `# 标题:` / `# 来源:` / `# 权威机构:` / `# 令号:` / `# 施行日期:` / `# 效力状态:` / `# 排序键:` 等 front-matter 行，避免噪声进入 embedding；同时保留为 chunk 元数据，供引用溯源与检索权威度过滤。
- **多级降级**：Markdown 标题切分 → 法律条款切分 → 空行段落切分；相邻小片段（< 80 字）自动合并，减少向量编码次数。
- **多格式兼容**：同一套规则兼容「每行一条」与「整页挤成一行表格」等多种法规排版，并支持 PDF / Word / Markdown。

> 对应实现：`backend/utils/document_parser.py`（`parse_markdown` / `extract_front_matter`）与 `backend/utils/financial_document_parser.py`；设计决策见项目 `architecture.md` 的 **ADR-009**。独立爬虫产线 `finance_rag_data/chunk_regulations.py` 采用相同边界感知规则，输出 `chunks.jsonl`。

---

## 三层幻觉抑制与时效风险提醒

金融 RAG 最怕"答错了还像真的"——模型引用已废止/失效条款、混用不同版本、或编造条文编号。系统在检索→生成→生成后三个位置各设一道防线，这是区别于普通聊天机器人的核心合规设计：

- **L1 检索端权威度过滤**（`hallucination_guard.filter_by_authority`）：在检索结果喂给 LLM 前，根据每个 chunk 的元数据（`权威机构` / `效力状态` / `施行日期` / `排序键` / `令号`）剔除 `已废止/失效` 候选、同法规多版本去重保最新、按权威机构权重排序。
- **L2 Prompt 强约束**（`hallucination_guard.build_constrained_system_note`）：在 LLM 的 system prompt 硬性注入——只能依据检索片段、无依据须答"未找到相关信息"、严禁编造条款/编号、发现已废止条款须提示时效风险。
- **L3 生成后校验**（`hallucination_guard.verify_answer`）：答案生成后比对 `superseded.json`，命中已废止条款则打标（`QueryResponse.guard`），并以"不可引用已废止条款"的强约束重答一次兜底。

**前端时效风险提醒**：命中已废止条款时，`frontend/app_jinyu.py` 在答案区上方渲染 ⚠️ 琥珀色提醒条（由 `frontend/guard_banner.py` 的纯函数 `build_guard_banner_html` 生成），逐条展示：

- 📅 发行日期（失效法规的发行/施行日）
- 📜 明文废止依据（哪个决定/法规明文废止了该条款）
- 📅 废止日期（**决定施行日**口径）
- 🔗 查看官方原文（指向官方公报的链接）

> 对应实现与决策：设计决策见项目 `architecture.md` 的 **ADR-010**。关键文件：`backend/utils/hallucination_guard.py`（守卫）、`backend/utils/superseded.json`（已废止条款对照表，v3，当前含 5 条典型废止条款）、`backend/services/qa_service.py`（L1/L3 接线）、`backend/services/llm_service.py`（L2 接线）、`frontend/guard_banner.py`（提醒条渲染）。守卫模块为纯新增、零依赖，配有 `tests/test_hallucination_guard.py`（三层功能）、`tests/test_guard_wiring.py`（接线集成）、`tests/test_guard_banner.py`（提醒条）共 24 项测试。

> **测试样例**：`sample_test_docs/` 下提供 5 个 Markdown 样例（存贷比旧规 / 保本理财旧规 / 核准制 IPO 旧规 / 存款利率管制旧规 / 现行有效流动性监管），前 4 个上传后应触发 ⚠️ 提醒，最后 1 个用于验证正常问答不误报。

---

## 技术栈

### 前端
- **Streamlit** — 快速构建数据应用 UI

### 后端
- **FastAPI** — 高性能异步 Web 框架
- **Uvicorn** — ASGI 服务器

### 检索与 AI
| 组件 | 技术 | 用途 |
|------|------|------|
| 向量数据库 | ChromaDB | 嵌入向量持久化存储与相似度检索 |
| 嵌入模型 | BAAI/bge-large-zh-v1.5 | 中文文本嵌入（768维） |
| 重排序模型 | BAAI/bge-reranker-base | CrossEncoder 候选文档精排 |
| 关键词检索 | rank-bm25 (BM25Okapi) | 概率模型全文检索 |
| 中文分词 | jieba | 中文文本分词 |
| 大语言模型 | DeepSeek / OpenAI 兼容 API | 答案生成 |
| 图数据库 | Neo4j（可选） | 金融知识图谱 |

### 文档解析
- **PyMuPDF (fitz)** — PDF 文档解析
- **python-docx** — Word 文档解析

### 开发与运维
- **LangSmith** — 全链路追踪（可选）
- **RAGAS** — RAG 自动化评估
- **Docker** — 容器化部署

---

## API 文档

后端运行后，访问 `http://localhost:8006/docs` 获取 Swagger UI 交互式文档。

### 核心接口

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查，返回系统状态 |
| `/query` | POST | 问答查询 |
| `/upload` | POST | 同步上传文档 |
| `/upload/start` | POST | 异步上传启动 |
| `/upload/progress/{task_id}` | GET | 异步上传进度轮询 |
| `/documents` | GET | 文档列表 |
| `/documents/{id}` | DELETE | 删除文档 |
| `/documents/{id}/file` | GET | 下载原始文件 |
| `/documents/{id}/view` | GET | HTML 文档预览 |

### 查询请求格式

```json
{
  "question": "反洗钱法第58条是什么",
  "top_k": 5,
  "use_rerank": true,
  "use_keyword_search": true
}
```

### 查询响应格式

```json
{
  "answer": "根据《中华人民共和国反洗钱法》第五十八条规定...",
  "citations": [
    {
      "document_name": "反洗钱法.pdf",
      "content": "第五十八条...",
      "score": 0.9971,
      "page_number": 15,
      "paragraph_number": 58
    }
  ],
  "found_in_knowledge_base": true,
  "processing_time_ms": 1234
}
```

---

## 配置说明

复制 `.env.example` 为 `.env`，按需修改以下配置：

```ini
# API 配置
API_HOST=0.0.0.0
API_PORT=8006

# LLM 配置（DeepSeek / OpenAI 兼容 API）
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=deepseek-chat

# 模型配置
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
RERANK_MODEL=BAAI/bge-reranker-base

# ChromaDB 配置
CHROMA_PERSIST_DIR=./backend/data/chroma

# 图数据库（可选）
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=password

# 追踪（可选）
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=your_langsmith_key
```

> **注意**：`OPENAI_API_KEY` 是必填项，系统通过该 Key 调用 LLM 生成答案。可使用 DeepSeek（`https://api.deepseek.com`）或其他兼容 API 服务。

---

## 模型下载

首次运行必须下载嵌入模型 `BAAI/bge-large-zh-v1.5`（约 1.3GB）。

```bash
python download_model.py
```

脚本会自动从 `hf-mirror.com` 镜像下载。如遇网络问题，可尝试：

1. 手动设置代理：
   ```bash
   set HF_ENDPOINT=https://hf-mirror.com
   python download_model.py
   ```

2. 将已下载的模型文件夹复制到 `backend/data/model_cache/` 目录

3. 系统会在不可用时自动降级到轻量模型 `paraphrase-multilingual-MiniLM-L12-v2`

---

## 贡献者规范

### 贡献流程

1. **Fork 项目** → 创建特性分支 `feature/your-feature-name`
2. **开发**：遵循现有代码风格，保持单文件内聚
3. **提交信息**：使用中文，格式为 `类型: 简短描述`，如：
   - `修复: 端口检测时正则匹配不精确的问题`
   - `新增: 异步上传进度轮询功能`
   - `优化: 重构条款号归一化逻辑`
4. **测试**：确保现有单元测试通过
5. **Pull Request**：描述变更内容和测试结果

### 代码规范

- **Python**：遵循 PEP 8，使用 4 空格缩进
- **文档注释**：类和方法使用中文字段说明
- **决策记录**：重要设计决策文件头标注 `决策记录` 区块
- **单文件风格**：前端页面 CSS 和 JS 内联于 `.py` 文件

### 行为准则

- 所有代码贡献需通过审查
- 不得引入与金融/法律问答无关的依赖
- 保持"拒绝幻觉"原则，任何检索不到的场景都应明确告知

---

## 许可证

本项目采用 **MIT License**。

```
MIT License

Copyright (c) 2026 JinYu AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 问题反馈

### 提交 Issue

如遇到问题或需要新功能，请通过以下方式反馈：

1. **GitHub Issues**：提交详细的 Bug 报告或功能请求
2. **反馈内容模板**：
   ```
   ## 问题描述
   [清晰简洁地描述问题]
   
   ## 复现步骤
   1. [第一步]
   2. [第二步]
   3. ...
   
   ## 预期行为
   [描述应该发生什么]
   
   ## 实际行为
   [描述实际发生了什么]
   
   ## 环境信息
   - 操作系统：[Windows/macOS/Linux]
   - Python 版本：[如 3.11.0]
   - 浏览器：[如 Chrome 120]
   ```

### 常见问题

**Q：首次启动报错"模型加载失败"？**  
A：请先运行 `python download_model.py` 下载嵌入模型。如网络受限，确保已设置 `HF_ENDPOINT=https://hf-mirror.com`。

**Q：问答返回"未找到相关信息"？**  
A：请先上传文档，系统仅基于已上传的知识库内容回答问题。

**Q：后端启动提示"端口被占用"？**  
A：运行 `python restart_backend.py` 自动清理，或手动终止占用进程。默认后端端口 8006，前端端口 8506。

**Q：启动时提示编码错误？**  
A：Windows 用户若遇到 GBK 编码错误，请在终端执行 `chcp 65001` 切换为 UTF-8 编码。

---

> **金语AI Financial Intelligence System**  
> 金语 = 金融智慧之语  
> 让每一份专业知识都可检索、可追溯、可信赖。
