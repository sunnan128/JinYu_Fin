# 金语AI 金融知识问答系统 —— 架构设计文档

> **文档版本**：v1.1  
> **最后更新**：2026-07-24  
> **文档状态**：已评审定稿（2026-07 补充 ADR-009 切块边界感知修复）

---

## 目录

1. [概述](#1-概述)
2. [技术栈选型](#2-技术栈选型)
3. [总体架构设计](#3-总体架构设计)
4. [分层架构详解](#4-分层架构详解)
5. [核心模块依赖关系](#5-核心模块依赖关系)
6. [数据流转逻辑](#6-数据流转逻辑)
7. [部署拓扑结构](#7-部署拓扑结构)
8. [第三方服务集成方案](#8-第三方服务集成方案)
9. [架构决策记录（ADR）](#9-架构决策记录adr)
10. [性能指标要求](#10-性能指标要求)
11. [可扩展性设计说明](#11-可扩展性设计说明)
12. [附录](#12-附录)

---

## 1. 概述

### 1.1 项目定位

金语AI 是一套面向金融与法律领域的 **RAG（检索增强生成）知识问答系统**，以"检索可靠、答案可溯、拒绝幻觉"为设计原则，为金融从业者、法律工作者提供基于专业文档的智能问答能力。

### 1.2 核心能力

| 能力 | 说明 |
|------|------|
| 语义检索 | 基于向量嵌入的语义相似度搜索 |
| 关键词检索 | 基于 BM25 的全文检索，支持中文分词 |
| 重排序精排 | 使用 CrossEncoder 对候选文档重排序 |
| 混合检索融合 | RRF（Reciprocal Rank Fusion）多路召回融合 |
| 层级关联评分 | 法律条款号层级关联匹配增强 |
| 知识图谱检索 | 金融实体关系图谱（Neo4j / Mock 降级） |
| 文档解析 | PDF / Word / Markdown 多格式支持 |
| 异步上传 | 大文档异步处理 + 进度轮询 |
| 溯源输出 | 回答附带引用来源（文档名/章节/条款/页码） |

### 1.3 设计原则

- **拒绝幻觉**：未在知识库中检索到相关内容时，明确告知用户"未找到相关信息"
- **溯源可信**：每个回答均附带引用来源，支持点击跳转至源文档
- **渐进增强**：从语义检索 → 重排序 → 图谱检索，逐层增强检索精度
- **可降级**：任一组件不可用（Neo4j / LLM / Reranker）时，系统自动降级运行
- **金融合规**：常驻免责声明，明确告知系统非专业金融建议提供方

---

## 2. 技术栈选型

### 2.1 技术栈一览

| 层级 | 技术组件 | 版本 | 选型理由 |
|------|----------|------|----------|
| **后端框架** | FastAPI | 0.109.0 | 异步支持、自动 OpenAPI 文档、类型安全 |
| **ASGI 服务器** | Uvicorn | 0.27.0 | 轻量高性能 ASGI 服务器 |
| **前端框架** | Streamlit | 1.29.0 | 快速构建数据应用 UI，Python 原生 |
| **向量数据库** | ChromaDB | 0.4.22 | 嵌入式持久化，无需独立部署服务 |
| **嵌入模型** | BAAI/bge-large-zh-v1.5 | — | 中文语义嵌入 SOTA，768 维向量 |
| **重排序模型** | BAAI/bge-reranker-base | — | CrossEncoder 精排，输出 Logits 分数 |
| **关键词检索** | rank-bm25 (BM25Okapi) | 0.2.2 | 经典概率检索模型，确定性可解释 |
| **中文分词** | jieba | — | 轻量高效中文分词 |
| **LLM** | OpenAI / DeepSeek 兼容 API | — | 兼容 OpenAI 协议，可替换任意 LLM |
| **图数据库** | Neo4j（可选） | — | 可选依赖，不可用时降级 Mock |
| **文档解析** | PyMuPDF / python-docx | — | PDF 和 Word 解析 |
| **追踪** | LangSmith（可选） | ≥0.1.0 | 全链路追踪，通过环境变量开关 |
| **评估** | RAGAS | ≥0.1.0 | RAG 自动化评估框架 |

### 2.2 关键依赖

```text
fastapi==0.109.0        uvicorn[standard]==0.27.0
streamlit==1.29.0       chromadb==0.4.22
sentence-transformers==2.3.1   rank-bm25==0.2.2
pymupdf==1.23.8         python-docx==1.1.0
openai==1.10.0          pydantic==2.5.3
jieba                   numpy==1.26.3
neo4j                    # 可选
langsmith>=0.1.0         # 可选
ragas>=0.1.0             # 评估用
```

---

## 3. 总体架构设计

### 3.1 架构分层

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Presentation Layer                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Streamlit 前端 (app_jinyu.py)                    │  │
│  │  文档管理 Tab · 智能问答 Tab · 免责声明 · 系统状态           │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                              │ HTTP REST                            │
├──────────────────────────────┼──────────────────────────────────────┤
│                    API Layer (FastAPI)                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  /health  /query  /upload  /upload/start  /upload/progress   │  │
│  │  /documents  /documents/{id}  /documents/{id}/view           │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
├──────────────────────────────┼──────────────────────────────────────┤
│                     Service Layer                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │  │
│  │  │ QA Service  │  │ LLM Service  │  │ Hybrid Retriever │   │  │
│  │  │ (路由/调度)  │  │ (重排/生成)   │  │ (三路混合检索器)   │   │  │
│  │  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘   │  │
│  │         │                │                    │              │  │
│  │  ┌──────┴──────────────────┴────────────────────┴─────────┐  │  │
│  │  │           Vector Store Service                         │  │  │
│  │  │  语义搜索 · BM25关键词 · RRF融合 · 层级关联评分         │  │  │
│  │  └───────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────┐  ┌──────────────┐                    │  │
│  │  │  Graph Service   │  │Entity Linker │                    │  │
│  │  │ (Neo4j/Mock)     │  │(金融实体链接)  │                    │  │
│  │  └──────────────────┘  └──────────────┘                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
├──────────────────────────────┼──────────────────────────────────────┤
│                     Data / Storage Layer                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌───────────┐ │  │
│  │  │ ChromaDB │  │ BM25 Pickle│  │ 文件系统  │  │ Model Cache│ │  │
│  │  │(向量存储)  │  │(关键词索引)  │  │(上传文档)  │  │(嵌入模型)   │ │  │
│  │  └──────────┘  └────────────┘  └──────────┘  └───────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件职责

| 组件 | 职责 |
|------|------|
| **QA Service** | 查询路由中枢，协调语义/关键词/重排序三种路径，处理降级和保底逻辑 |
| **Vector Store Service** | 向量存储管理，嵌入编码，语义搜索，BM25 索引构建与检索，RRF 融合，层级关联评分 |
| **LLM Service** | CrossEncoder 重排序，LLM 答案生成，调用 DeepSeek/OpenAI 兼容 API |
| **Hybrid Retriever** | 三路召回（BM25 + Vector + Graph）+ RRF 融合 + CrossEncoder 重排 |
| **Entity Linker** | 金融实体别名匹配，从问句抽取标准名 + 行业编码 |
| **Graph Service** | 金融知识图谱查询（Neo4j Cypher），不可用时降级到内置 Mock 图谱 |
| **Document Parser** | 多格式文档解析 + 切分策略（标题切分/**边界感知**条款切分/段落切分） |

---

## 4. 分层架构详解

### 4.1 Presentation Layer（表示层）

前端为 Streamlit 单页应用，采用**深绿色品牌配色**。页面结构：

```
┌─ 侧边栏 ──────────────────────────────────────┐
│  品牌标识 · 系统状态指示器 · 重连按钮           │
│  关于系统说明 · 技术支持信息                    │
├─ 主内容区域 ────────────────────────────────────┤
│  ┌─ 免责声明横幅 ───────────────────────────┐  │
│  │  常驻"非金融建议"声明                    │  │
│  ├─ Tab: 文档管理 ──────────────────────────┤  │
│  │  文件上传(异步+进度条) · 已上传文档列表   │  │
│  │  删除文档                                │  │
│  ├─ Tab: 智能问答 ──────────────────────────┤  │
│  │  检索选项组(重排序/关键词) · 输入框       │  │
│  │  问答卡片 · 引用来源列表 · 处理时间标签   │  │
│  └── 页脚 ────────────────────────────────┘  │
└───────────────────────────────────────────────┘
```

**关键交互**：
- 异步上传：POST `/upload/start` → 每 0.5s 轮询 `/upload/progress/{task_id}` 直到 stage="done"
- 检索选项：复选框 `use_rerank` + `use_keyword_search`，组合出 4 种检索模式
- 重连按钮：通过 `subprocess` 执行 `restart_backend.py` 脚本

### 4.2 API Layer（API 层）

FastAPI 应用，CORS 全放开（`allow_origins=["*"]`），服务端前缀：`http://localhost:8006`

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查，返回 vector_db + LLM 状态 |
| `/query` | POST | 问答查询，接收 `QueryRequest`，返回 `QueryResponse` |
| `/upload` | POST | 同步上传（向后兼容） |
| `/upload/start` | POST | 异步上传启动，返回 `task_id` |
| `/upload/progress/{task_id}` | GET | 异步上传进度轮询 |
| `/documents` | GET | 获取文档列表 |
| `/documents/{id}` | DELETE | 删除文档 |
| `/documents/{id}/file` | GET | 下载原始文件 |
| `/documents/{id}/view` | GET | HTML 文档预览 |

### 4.3 Service Layer（服务层）

#### 4.3.1 QA Service（查询路由中枢）

`backend/services/qa_service.py`

```
用户查询
    │
    ├── 条款号归一化（"第35条" ↔ "第三十五条" 双向转换）
    │
    ├── [use_rerank=True]  ──────────────────────────────────────────┐
    │   ├── ① keyword_fallback: 强制 Hybrid Search (use_keyword=True)│
    │   │    （始终开启，作为保底降级）                                │
    │   ├── ② candidates: Hybrid Search (rerank_candidates=20,       │
    │   │    use_keyword=request.use_keyword_search)                 │
    │   ├── ③ CrossEncoder 精排 top_k                               │
    │   ├── ④ 安全兜底: search_results 为空 → keyword_fallback       │
    │   ├── ⑤ keyword_fallback 插队: 未在 rerank 结果中的 doc 插入   │
    │   │    （rerank_score = top_score × 0.995）                    │
    │   └── ⑥ 重排序 + 截断 top_k                                   │
    │                                                                 │
    ├── [use_keyword_search=True] → Hybrid Search (语义+关键词)       │
    │                                                                 │
    └── [else] → Semantic Search (纯语义)                            │
```

**关键路径**：

1. **rerank + keyword 模式**：keyword_fallback 强制全混合搜索 → rerank 大候选池 → CrossEncoder 精排 → 合并 keyword_fallback 缺失文档 → 重排序 → top_k
2. **keyword-only 模式**：混合搜索（语义 + BM25 关键词）
3. **语义模式**：纯语义搜索

**降级策略**：当 CrossEncoder 分数全为 NaN/Inf 时，回退到混合检索原始排序。

#### 4.3.2 Vector Store Service（向量存储服务）

`backend/services/vector_store.py`

**嵌入模型加载策略（6 层降级）**：
1. 检查本地 HF Hub 缓存结构（snapshot）
2. 检查本地扁平缓存目录
3. 在线加载指定模型（`BAAI/bge-large-zh-v1.5`）
4. 轻量级备用模型（`paraphrase-multilingual-MiniLM-L12-v2`）
5. 失败则抛 `RuntimeError` 要求运行 `download_model.py`

**混合检索流程**：

```
用户查询
    │
    ├── _normalize_article_numbers()  # 条款号归一化（阿拉伯↔中文）
    │
    ├── semantic_search()  # ChromaDB 余弦相似度搜索
    │   └── 向量编码 → ChromaDB `query()` → 返回 top_k 结果
    │
    ├── keyword_search()   # BM25 关键词搜索
    │   └── jieba 分词 → BM25Okapi `get_scores()` → 排序 top_k
    │
    ├── RRF 融合 (K=20)
    │   └── RRF(d) = 1/(20 + rank_sem(d)) + 1/(20 + rank_kw(d))
    │
    └── 层级关联评分
        ├── 精确匹配法律名称: +0.03
        ├── 子关键词匹配: +0.02
        ├── 法律名称别名匹配: +0.01
        └── 条款号双向归一化匹配
```

**BM25 索引**：以 pickle 持久化到磁盘（`backend/data/bm25_index.pkl`），仅在导入模块时加载到内存，不常驻重建。

#### 4.3.3 LLM Service（大模型服务）

`backend/services/llm_service.py`

**重排序模块**：
- 使用 `BAAI/bge-reranker-base` CrossEncoder
- 输入：`[query, doc_content]` pairs
- 输出：Logits（未经 Sigmoid，可为负值）
- **重要设计决策**：移除硬阈值 0.1——bge-reranker-base 输出 Logits 范围可负，阈值 0.1 会误触发降级，替换为 NaN/Inf 检测

**答案生成**：
- 调用 OpenAI 兼容 API（DeepSeek / 任意兼容服务）
- 将检索结果拼入 Prompt 上下文
- 未检索到相关内容时返回"未找到相关信息"

#### 4.3.4 Hybrid Retriever（混合检索器）

`backend/services/hybrid_retriever.py`

```
用户问句 → Entity Linker (实体抽取)
              │
      ┌───────┼───────┐
      ↓       ↓       ↓
   BM25   Vector   Graph (Neo4j)
      │       │       │
      └───────┼───────┘
              ↓
      RRF(K=20) 融合
              ↓
     CrossEncoder 重排
              ↓
      ┌───────┴───────┐
      ↓               ↓
  置信度≥0.1       置信度<0.1
      │               │
      ↓               ↓
  精排 Top-5 ← 回退至 Vector Top-5
```

#### 4.3.5 Entity Linker（实体链接器）

`backend/utils/entity_linker.py`

- 内置 30+ 金融实体别名表（公司/产品/法规/概念/指标）
- 三层匹配：精确匹配 → 同义匹配 → 上下文短语匹配
- 别名按长度降序排列，优先匹配长词
- 使用重叠区间检测避免重复匹配

#### 4.3.6 Graph Service（图谱服务）

`backend/services/graph_service.py`

**图结构**：
```
(Company)-[:BELONGS_TO]->(Industry)
(Company)-[:ISSUES]->(Product)
(Regulation)-[:GOVERNS]->(Industry)
(Company)-[:RELATED_TO]->(Concept)
```

- Neo4j 为可选依赖，不可用时降级到内置 Mock 图谱
- Mock 图谱包含 20+ 节点和 15+ 关系
- 输出格式：`{nodes, edges, text_summary}`

### 4.4 Data Layer（数据层）

| 数据存储 | 内容 | 持久化路径 |
|----------|------|-----------|
| ChromaDB | 文档向量 + 元数据 | `backend/data/chroma/` |
| BM25 Pickle | BM25 索引（文档原文） | `backend/data/bm25_index.pkl` |
| 文件系统 | 上传的原始文档 | `backend/data/uploads/` |
| Model Cache | 嵌入模型缓存 | `backend/data/model_cache/` |
| Neo4j（可选） | 金融知识图谱 | 外部数据库 |
| LangSmith（可选） | 全链路追踪 | SaaS |

---

## 5. 核心模块依赖关系

```
main.py (FastAPI)
  ├── config.py                  # 配置读取
  ├── models/schemas.py          # 数据模型
  └── services/qa_service.py     # 查询路由
        ├── services/vector_store.py   # 向量 + BM25 + RRF
        │     └── utils/document_parser.py  # 文档解析
        └── services/llm_service.py     # 重排 + LLM 生成

hybrid_retriever.py
  ├── services/vector_store.py
  ├── services/graph_service.py
  │     └── config.py
  └── utils/entity_linker.py
```

**初始化依赖顺序**：
1. `config.py` — 加载配置
2. `schemas.py` — 数据模型（无依赖）
3. `document_parser.py` — 文档解析工具（无依赖）
4. `entity_linker.py` — 实体链接（无依赖）
5. `graph_service.py` — 图谱服务（依赖 config）
6. `vector_store.py` — 向量存储（依赖 config, document_parser, ChromaDB, SentenceTransformer, jieba, BM25）
7. `llm_service.py` — LLM 服务（依赖 config, OpenAI, sentence-transformers）
8. `qa_service.py` — 查询路由（依赖 vector_store, llm_service）

---

## 6. 数据流转逻辑

### 6.1 文档上传流程

```
用户上传文件
    │
    ├── 同步流程 (/upload)
    │   └── 保存 → 解析 → 切分 → 向量编码 → ChromaDB 存储 → 更新 BM25
    │
    └── 异步流程 (/upload/start + /upload/progress/{id})
        ├── POST /upload/start → 返回 task_id
        ├── 后台线程：保存 → 解析 → 切分 → 编码 → 存储
        └── 前端轮询 GET /upload/progress/{id} 获取进度
```

**文档切分策略**（3级降级）：
1. Markdown 标题切分（一级/二级标题）
2. 法律条款切分（**边界感知**切分："第X条"正则 + 前置边界判断）
3. 空行段落切分

> **边界感知切分说明（2026-07 修订）**：法律条款切分不再简单按"第X条"字面 split，而是仅在「句末标点（。；！？）/ 换行 / 全角空格（`\u3000`）」之后才识别为新法条起点；"本法第五十三条""依照本法第五十二条至第五十四条规定"等**行内引用**不会被误切为独立块，确保一条法条（如《反洗钱法》第五十五条、第五十六条）完整成块。同时剔除各块内的 `# 标题:` / `# 来源:` 元信息噪声行，避免污染 embedding 与检索。详见 [ADR-009](#adr-009法律条款边界感知切分)。

**片段合并**：`MIN_CHUNK_SIZE=80`，相邻小片段自动合并，减少向量编码次数。

### 6.2 问答流程

```
用户提问
    │
    ├── 1. 条款号归一化（"第35条" ↔ "第三十五条"）
    │
    ├── 2. 检索选项判断
    │   ├── Rerank + Keyword → 三路混合 + 重排序 + 关键词保底
    │   ├── Keyword Only → 语义 + BM25 混合
    │   └── 纯语义 → ChromaDB 语义搜索
    │
    ├── 3. 层级关联评分（按法律名称匹配加分）
    │
    ├── 4. LLM 答案生成（检索结果 → Prompt → API 调用）
    │
    └── 5. 返回结果（答案 + 引用来源 + 处理时间）
```

### 6.3 混合检索数据流（HybridRetriever）

```
                          ┌──────────────────────┐
                          │   用户问句 (query)    │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   Entity Linker      │
                          │   (金融实体抽取)       │
                          └──────────┬───────────┘
                                     │ entity_list
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
    ┌─────────▼──────────┐ ┌─────────▼──────────┐ ┌─────────▼──────────┐
    │   BM25 关键词召回   │ │  Vector 语义召回    │ │  Graph 图谱检索    │
    │  (rank_bm25+jieba) │ │  (ChromaDB+BGE)    │ │  (Neo4j/Mock)     │
    └─────────┬──────────┘ └─────────┬──────────┘ └─────────┬──────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   RRF 融合 (K=20)    │
                          │   1/(K+rank) 累加    │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  CrossEncoder 重排   │
                          │  (bge-reranker-base) │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  引用溯源 & 结果输出  │
                          │  (文档名/条款/页码)   │
                          └──────────────────────┘
```

---

## 7. 部署拓扑结构

### 7.1 开发/本地部署

```
┌──────────────────────────────────────────────┐
│              本地单机部署                      │
│                                               │
│  ┌──────────────┐    ┌──────────────────┐    │
│  │  FastAPI      │    │  Streamlit       │    │
│  │  :8006        │◄───│  :8506           │    │
│  └──────┬───────┘    └──────────────────┘    │
│         │                                     │
│  ┌──────┴─────────────────────────────┐      │
│  │  数据目录 backend/data/             │      │
│  │  ├── chroma/ (ChromaDB 持久化)     │      │
│  │  ├── model_cache/ (模型缓存)        │      │
│  │  ├── uploads/ (上传文档)            │      │
│  │  └── bm25_index.pkl (BM25 索引)    │      │
│  └────────────────────────────────────┘      │
│                                               │
│  启动方式: python start_jinyu.bat             │
│  或: uvicorn backend.main:app --port 8006    │
│      streamlit run frontend/app_jinyu.py     │
└──────────────────────────────────────────────┘
```

### 7.2 Docker 部署

```yaml
version: '3.8'
services:
  backend:
    build: .
    container_name: jinyu-backend
    ports: ["8000:8000"]
    environment:
      - API_PORT=8000
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./backend/data:/app/backend/data
  frontend:
    build: .
    container_name: jinyu-frontend
    ports: ["8501:8501"]
    depends_on: [backend]
    command: streamlit run frontend/app.py --server.port 8501
```

### 7.3 网络拓扑

```
               ┌──────────┐
               │ 浏览器    │
               │(用户访问) │
               └─────┬────┘
                     │ HTTP
          ┌──────────┴──────────┐
          │                     │
  ┌───────▼───────┐   ┌────────▼───────┐
  │ FastAPI       │   │ Streamlit      │
  │ 后端 :8006    │   │ 前端 :8506     │
  └───────┬───────┘   └────────────────┘
          │
  ┌───────┴───────────────────────────┐
  │   外部服务                        │
  │  ├── OpenAI / DeepSeek API       │
  │  ├── Hugging Face (模型下载)      │
  │  ├── ModelScope (模型下载备用)    │
  │  └── LangSmith (追踪)            │
  └──────────────────────────────────┘
```

---

## 8. 第三方服务集成方案

### 8.1 LLM 服务（OpenAI 兼容 API）

```python
# config.py
LLM_MODEL = "deepseek-chat"           # 可更换为 gpt-4, qwen 等
OPENAI_BASE_URL = "https://api.openai.com/v1"  # 可更换为任何兼容 API
```

- 通过 OpenAI SDK 调用，`base_url` 可指向任意 OpenAI 兼容服务
- 环境变量控制 API Key：`OPENAI_API_KEY`
- 默认模型：`deepseek-chat`，配置在 `.env` 中

### 8.2 Hugging Face 模型服务

| 模型 | 用途 | 加载方式 | 缓存策略 |
|------|------|---------|---------|
| BAAI/bge-large-zh-v1.5 | 文本嵌入（768 维） | SentenceTransformer | `model_cache/` 持久化 |
| BAAI/bge-reranker-base | CrossEncoder 重排序 | 懒加载（首次查询时） | `model_cache/` 持久化 |

**模型下载策略**：
1. 优先检查本地 HF Hub 缓存结构
2. 通过 `hf-mirror.com` 镜像下载（环境变量 `HF_ENDPOINT`）
3. 降级到 `paraphrase-multilingual-MiniLM-L12-v2`
4. 最后通过 `download_model.py` 脚本手动下载

### 8.3 Neo4j 图数据库

```python
# config.py（可选配置）
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
```

- **可选依赖**：Neo4j 不可用时自动降级到内置 Mock 图谱
- Mock 图谱包含 20+ 节点（公司/行业/产品/法规/概念）
- 15+ 关系（BELONGS_TO / ISSUES / GOVERNS / RELATED_TO）

### 8.4 LangSmith 追踪

```python
# config.py（可选配置）
LANGSMITH_TRACING = False  # 默认关闭
LANGSMITH_API_KEY = ""
LANGSMITH_PROJECT = "jinyu-rag"
```

- 通过 `traceable` 装饰器实现
- 关闭时使用 Noop 装饰器，零额外开销
- 通过环境变量 `LANGSMITH_TRACING=true` 开启

---

## 9. 架构决策记录（ADR）

### ADR-001：混合检索采用 RRF 而非加权求和

- **状态**：已实施
- **背景**：初始方案采用 `semantic_weight=0.7 + keyword_weight=0.3` 加权求和
- **决策**：改用 Reciprocal Rank Fusion（K=20），融合分数 = Σ 1/(K+rank)
- **理由**：RRF 不依赖分数值的量纲对齐，语义和 BM25 分数范围差异大时加权求和不稳定
- **影响**：语义和关键词分数各自独立，融合结果更稳定。文档 `vector_store.py` 中 RRF 实现在 qa_service 调用路径中，加权求和在 vector_store.hybrid_search 中保留

### ADR-002：BM25 索引持久化到磁盘，不在 RAM 常驻

- **状态**：已实施
- **背景**：每次启动时重新遍历 ChromaDB 构建 BM25 索引，大文档集合构建耗时
- **决策**：首次构建后序列化到 `backend/data/bm25_index.pkl`，启动时优先从磁盘加载
- **理由**：避免每次启动的 BM25 重建开销，减少冷启动时间
- **影响**：新增文档后自动触发 BM25 重建并持久化

### ADR-003：层级关联评分解决条款号不匹配

- **状态**：已实施
- **背景**：法律条款块正文（如"第三十五条…"）不含法律名称（如"民法典"），纯语义/关键词检索无法匹配合适的 top-20 候选池
- **决策**：实现 `_parse_hierarchical_query` + 元数据补充检索 + `_apply_hierarchical_scoring`
- **理由**：利用用户问句中的法律名称信息，在元数据维度补充检索，并在评分阶段按匹配程度加分
- **影响**：法律领域检索召回率显著提升

### ADR-004：移除 Rerank 置信度硬阈值

- **状态**：已修订（2026-07）
- **背景**：bge-reranker-base 输出未归一化 Logits（范围可负），硬阈值 0.1 导致正确结果被误判为低分而降级
- **决策**：移除 `RERANK_CONFIDENCE_THRESHOLD=0.1`，替换为 NaN/Inf 检测
- **理由**：Logits 分数绝对值无意义，仅排序有意义；阈值 0.1 是因果用中 0~1 概率思维的错误假设
- **影响**：重排序路径和关键词路径结果保持一致

### ADR-005：关键词保底降级机制

- **状态**：已实施（2026-07）
- **背景**：Rerank 路径中，若 BM25 参数 `use_keyword=False`，rerank 后的 top-5 可能缺失目标文档
- **决策**：在 rerank 路径中，始终执行一次完整混合搜索作为 `keyword_fallback`，并将其中未出现在 rerank 结果中的文档插入最终结果（`rerank_score = top_score × 0.995`）
- **理由**：确保 rerank 路径不丢失关键词能够匹配到的文档
- **影响**：rerank 路径召回率 100% 对齐 keyword-only 路径

### ADR-006：异步上传 + 进度轮询

- **状态**：已实施
- **背景**：大文档（如整部法律 PDF）解析 + 编码时间较长，同步上传阻塞前端
- **决策**：新增 `/upload/start` 和 `/upload/progress/{id}` 端点，后台线程处理
- **理由**：避免前端请求超时，提供实时进度反馈
- **影响**：留存同步 `/upload` 端点以保持向后兼容

### ADR-007：Neo4j 为可选依赖，Mock 降级

- **状态**：已实施
- **背景**：知识图谱能增强金融实体关联检索，但部署 Neo4j 增加运维复杂度
- **决策**：Neo4j 作为可选依赖，连接失败时降级到内置 Mock 图谱
- **理由**：降低部署门槛，同时保留图谱检索能力
- **影响**：HybridRetriever 的 graph 路在无 Neo4j 时从 Mock 图谱检索

### ADR-008：条款号归一化双向转换

- **状态**：已实施
- **背景**：用户输入"第35条"（阿拉伯数字），文档写"第三十五条"（中文数字），BM25 分词后 token 完全不匹配
- **决策**：在 `qa_service.py`（查询侧）和 `vector_store.py`（评分侧）均实现阿拉伯↔中文数字双向转换
- **理由**：无论用户输入哪种格式、文档使用哪种格式，都能匹配
- **影响**：条款号检索准确率大幅提升

### ADR-009：法律条款边界感知切分

- **状态**：已实施（2026-07，对应 `backend/utils/document_parser.py` 与 `backend/utils/financial_document_parser.py` 修订）
- **背景**：原切分逻辑用 `re.split(r'(第[...]+条)', ...)` 在每一个"第X条"处都切开，导致法条正文中的行内引用（如"违反本法第五十三条、第五十四条""依照本法第五十二条至第五十四条规定"）被误判为新法条起点，一条完整法条（例如《反洗钱法》第五十五条、第五十六条）被切碎成多块，检索召回质量下降、引用溯源失真。
- **决策**：改为**边界感知切分**——识别"第X条"为新法条起点时，要求其前一个字符为句末标点（。；！？）、换行或全角空格（`\u3000`）/右括号；否则视为行内引用，保留在原块内。同时通过 `extract_front_matter` 将每个文档块顶部的全部 front-matter 元信息行（`# 标题:`/`# 来源:`/`# 权威机构:`/`# 令号:`/`# 施行日期:`/`# 效力状态:`/`# 排序键:`）从 chunk 正文中剔除，避免噪声进入 embedding 与检索，并保留为 chunk 元数据供 L1 权威度/时效过滤；独立爬虫产线 `finance_rag_data/chunk_regulations.py` 复用同一套边界感知规则，输出 `chunks.jsonl`。
- **理由**：中文法规层级为「章 → 条 → 款 → 项」，应以"条"为最小 chunk 单位；行内引用不应触发新块。边界字符判断可同时兼容"每行一条"与"整页挤成一行表格"等多种排版。
- **影响**：验证显示《反洗钱法》第五十三~五十七条均为完整独立块（约 230~456 字），行内引用不再被切断；元数据噪声行不再泄漏进向量库。

### ADR-010：三层幻觉抑制守卫接入 QA 主链路

- **状态**：已实施（2026-07，对应 `backend/utils/hallucination_guard.py` 与 `backend/services/qa_service.py` / `backend/services/llm_service.py` 修订）
- **背景**：金融 RAG 最怕"答错了还像真的"——模型引用已废止/失效条款、混用不同版本、或编造条文编号。普通聊天机器人"能聊"不够，金融场景必须答得准、答得新、答得有出处。
- **决策**：将 `hallucination_guard.py` 的三层守卫正式接入 `QAService.query()` 主链路（零改动检索/生成/返回既有逻辑，仅以包裹方式增强）：
  - **L1 检索端权威度过滤**：在 `query()` 检索结果出来后、喂给 LLM 前调用 `filter_by_authority(search_results)`——剔除 `效力状态=已废止/失效` 的候选块、同法规多版本按 `排序键/施行日期` 去重保最新版、按权威机构权重排序。修复了去重 bug（单文件多 chunk 不再被压成 1 块）。
  - **L2 Prompt 强约束**：在 `llm_service.generate_answer` 的 system prompt 注入 `build_constrained_system_note()`——仅依据检索片段、无依据必须答"未找到相关信息"、严禁编造条款/编号、发现已废止条款须提示时效风险。
  - **L3 生成后校验**：在 `generate_answer` 返回后调用 `verify_answer(answer)`，命中已废止条款则 `blocked=True` 并打标（写入 `QueryResponse.guard`）；随即以"不可引用已废止条款"的强约束重答一次（regenerate），重答干净则采用，否则保留原答案仍打标。这是最后一道兜底。
  - **数据底座**：异步上传路径 `_process_upload_background` 现把 chunk 的 front-matter 元数据（`权威机构/效力状态/施行日期/排序键/令号`）随同入库，使 L1 在真实异步上传链路中也能消费（此前该路径漏存元数据）；同步 `add_documents` 路径原本已携带。
  - **鲁棒性**：L1/L3 均包在 `try/except` 内，异常时降级为"不过滤/不拦截"，不影响正常问答；`superseded.json` 缺失时 L3 自动放行。
- **理由**：三层分别堵在"检索端 / 生成端 / 生成后"三个位置，是工业级 RAG 防伪的标准做法；守卫为纯新增独立模块，可独立测试，不影响既有功能。
- **影响**：新增 `tests/test_guard_wiring.py` 集成测试（上传/解析/检索/L1 拦截/L3 拦截-打标-重答 全链路，无需真实模型与 API）；连同 `tests/test_hallucination_guard.py` 共 18 项测试全部通过。`QueryResponse` 新增可选 `guard` 字段，前端可据此展示时效风险告警（旧前端忽略该字段，向后兼容）。
  - **前端展示（补充）**：`frontend/app_jinyu.py` 在答案区上方新增 ⚠️ 时效风险 提示条——仅当 `guard.blocked=True` 时渲染（`build_guard_banner_html()` 纯函数，位于新建 `frontend/guard_banner.py`，无 streamlit 依赖、可单测）。提示条列出命中的已废止/失效条款（法规名+条款+命中关键词+现行有效替代），告知用户系统已自动时效校验并尽量重答。含 `tests/test_guard_banner.py`（4 项）共 22 项测试通过。

---

## 10. 性能指标要求

### 10.1 响应时间

| 场景 | 目标 | 测量方法 |
|------|------|---------|
| 文档上传 + 解析 | < 30s（100 页 PDF） | 端到端计时 |
| 语义检索（top-5） | < 500ms | `/query` 端点计时 |
| 混合检索 + RRF | < 800ms | `/query` 端点计时 |
| 混合 + Rerank（top-5） | < 2000ms | `/query` 端点计时 |
| LLM 答案生成 | < 10000ms | LLM API 调用计时 |
| 前端页面加载 | < 3s | 浏览器 DevTools |
| 异步上传轮询间隔 | 500ms | 前端代码配置 |

### 10.2 准确率

| 指标 | 目标 | 说明 |
|------|------|------|
| 语义检索 Recall@5 | ≥ 85% | 相关性判定 |
| 混合检索 Recall@5 | ≥ 92% | 含关键词和语义 |
| Rerank NDCG@5 | ≥ 0.88 | 归一化贴现累积增益 |
| LLM 答案准确率 | ≥ 90% | 以 ground truth 为基准 |
| 拒绝幻觉率 | 100% | 未检索到时不生成虚假答案 |

### 10.3 容量

| 指标 | 限制 | 说明 |
|------|------|------|
| 单文件大小 | ≤ 10MB | 配置 `MAX_UPLOAD_SIZE` |
| ChromaDB 文档数 | 无硬限制 | 受磁盘容量限制 |
| BM25 索引大小 | 取决于文档数 | pickle 持久化 |
| 模型缓存 | ~2.5GB | bge-large + bge-reranker |
| 并发请求 | 无硬限制 | FastAPI 异步处理 |

---

## 11. 可扩展性设计说明

### 11.1 检索策略扩展

检索路径在 `qa_service.py` 中以 `if/elif/else` 清晰分支，新增检索策略只需：

1. 在 `models/schemas.py` 的 `QueryRequest` 中添加布尔字段
2. 在 `qa_service.py` 的 `query()` 方法中添加对应的 elif 分支
3. 前端新增复选框控件

### 11.2 向量数据库替换

当前使用 ChromaDB，方案通过 `VectorStoreService` 封装。替换为 Milvus / Pinecone / Qdrant：

1. 在 `vector_store.py` 中实现相同的公有接口（`semantic_search` / `hybrid_search` / `add_documents` / `delete_document` 等）
2. 配置项已有 `VECTOR_DB_TYPE=chroma | milvus`

### 11.3 嵌入模型替换

嵌入模型通过 `config.EMBEDDING_MODEL` 配置：
- 加载策略在 `_load_embedding_model()` 方法中统一管理
- 模型缓存通过 `SENTENCE_TRANSFORMERS_HOME` 环境变量控制

### 11.4 LLM 提供商替换

LLM 调用使用 OpenAI 兼容 API：
- 修改 `OPENAI_BASE_URL` 指向任意兼容服务
- 修改 `LLM_MODEL` 为对应的模型名称
- Prompt 模板在 `llm_service.py` 中集中管理

### 11.5 前端界面扩展

Streamlit 前端以 Tab 组织，扩展新功能：
1. 添加新 Tab（如"知识图谱可视化"）
2. 在现有 Tab 中添加子功能

### 11.6 评估流水线

`eval/` 目录下的 RAGAS 评估流水线支持：
- 基准模式（无 Rerank）
- Rerank 模式
- 双模式对比
- 支持 A/B 测试：同一数据集、不同配置、可复现的量化结果

### 11.7 多前端支持

项目已包含三个前端变体：
- `app.py` — 默认前端
- `app_jinyu.py` — 金语品牌前端（主推）
- `app_xinglin.py` — 杏林品牌前端

新前端只需实现与后端 JSON API 对接即可。

---

## 12. 附录

### A. 项目目录结构

```
project_root/
├── backend/
│   ├── __init__.py
│   ├── config.py                 # 配置管理
│   ├── main.py                   # FastAPI 应用入口
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic 数据模型
│   ├── services/
│   │   ├── __init__.py
│   │   ├── qa_service.py         # 查询路由中枢
│   │   ├── vector_store.py       # 向量存储 + BM25 + RRF
│   │   ├── llm_service.py        # 重排序 + LLM 生成
│   │   ├── hybrid_retriever.py   # 三路混合检索器
│   │   └── graph_service.py      # 图谱检索 (Neo4j/Mock)
│   ├── utils/
│   │   ├── document_parser.py    # 文档解析与切分
│   │   └── entity_linker.py      # 金融实体链接
│   └── data/
│       ├── chroma/               # ChromaDB 持久化
│       ├── model_cache/          # 嵌入模型缓存
│       └── bm25_index.pkl        # BM25 索引
├── frontend/
│   ├── app.py                    # 默认前端
│   ├── app_jinyu.py              # 金语品牌前端（主推）
│   └── app_xinglin.py            # 杏林品牌前端
├── eval/
│   ├── dataset.json              # 评估数据集
│   ├── run_eval.py               # 评估执行脚本
│   ├── comparison.py             # 对比分析
│   └── results/                  # 评估结果
├── .env.example                  # 环境变量模板
├── .env                          # 环境变量（不上传）
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── download_model.py             # 模型下载工具
├── install_dependencies.py       # 依赖安装工具
├── restart_backend.py            # 后端重启脚本
└── start_jinyu.bat               # 一键启动脚本
```

### B. 配置参考

关键配置项（`backend/config.py`）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `API_HOST` | `0.0.0.0` | API 监听地址 |
| `API_PORT` | `8006` | API 端口 |
| `LLM_MODEL` | `deepseek-chat` | 大模型名称 |
| `EMBEDDING_MODEL` | `BAAI/bge-large-zh-v1.5` | 嵌入模型 |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | 重排序模型 |
| `RERANK_CANDIDATES` | `20` | 重排序候选数 |
| `CHROMA_PERSIST_DIR` | `./backend/data/chroma` | ChromaDB 路径 |
| `MAX_UPLOAD_SIZE` | `10485760` | 最大上传大小（10MB） |
| `LANGSMITH_TRACING` | `False` | LangSmith 追踪开关 |

### C. 端口规划

| 服务 | 默认端口 |
|------|---------|
| FastAPI 后端 | 8006 |
| Streamlit 前端 | 8506 |
| FastAPI OpenAPI Docs | 8006 (同后端，`/docs`) |

---

> **文档维护者**：金语AI 开发团队  
> **反馈渠道**：[项目 Issues](file:///d:/trae_project08_Jinyu)
