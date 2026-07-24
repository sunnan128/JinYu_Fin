# ── 数据模型（Pydantic v2） ──
# 决策记录：
# - 使用 Pydantic v2（FastAPI 内置），自动校验 + 生成 OpenAPI 文档
# - Citation.paragraph_number 可为空：Word 文档段落号与 PDF 页码结构不同
# - QueryRequest.use_rerank / use_keyword_search 默认 True：法律场景需精确匹配 + 重排序
# - UploadResponse.chunk_count 返回片段数，前端据此判断入库是否成功
# - DocumentInfo.upload_time 用 datetime 而非 str：前端可格式化为本地时区

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentInfo(BaseModel):
    id: str
    filename: str
    upload_time: datetime
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None

class DocumentChunk(BaseModel):
    id: str
    document_id: str
    content: str
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_rerank: bool = True
    use_keyword_search: bool = True

class Citation(BaseModel):
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    content: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    found_in_knowledge_base: bool
    processing_time_ms: float

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    message: str
    chunk_count: int

class HealthResponse(BaseModel):
    status: str
    vector_db: str
    llm: str
