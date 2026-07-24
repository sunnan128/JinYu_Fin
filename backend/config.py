# ── 金语AI 金融文档智能问答系统 配置 ──
# 决策记录：
# - LLM: DeepSeek Chat (经 DeepSeek API 调用，兼容 OpenAI 格式)
# - 嵌入模型: BAAI/bge-large-zh-v1.5（已缓存至本地，优先 local_files_only 加载）
# - 向量库: ChromaDB（持久化到磁盘，重启不丢数据）
# - ChromaDB 持久化路径 ./backend/data/chroma：放在项目内便于打包迁移和文档同步
# - Milvus 配置保留但未使用：为未来分布式扩展预留接口，当前单机 ChromaDB 足够
# - 上传目录 ./backend/data/uploads 在 Settings 初始化时自动创建，无需手动建目录
# - 混合检索: RRF（Reciprocal Rank Fusion）替代加权求和，解决单路检索命中时被淹没的问题

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8006
    
    OPENAI_API_KEY: str = "demo_key"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "deepseek-chat"
    
    # 嵌入模型：可用 paraphrase-multilingual-MiniLM-L12-v2 替代以加快速度
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    RERANK_MODEL: str = "BAAI/bge-reranker-base"
    RERANK_CANDIDATES: int = 20  # 混合检索返回的候选数，供 rerank 精排后取 top_k
    
    VECTOR_DB_TYPE: str = "chroma"
    CHROMA_PERSIST_DIR: str = "./backend/data/chroma"
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    
    # Neo4j 图数据库（金融知识图谱）
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    SQLITE_DB_PATH: str = "./backend/data/legal_qa.db"
    
    SECRET_KEY: str = "legal-qa-system-secret-key-2024"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    UPLOAD_DIR: str = "./backend/data/uploads"
    MAX_UPLOAD_SIZE: int = 10485760
    
    # LangSmith 可观测性配置
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "default"
    LANGSMITH_TRACING: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.SQLITE_DB_PATH), exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
