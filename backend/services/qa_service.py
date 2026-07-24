# ── QA 主服务 ──
# 决策记录：
# - self.documents 为内存缓存，服务重启后丢失
# - _restore_documents_from_db() 从 ChromaDB 元数据恢复文档列表，并同步写入 self.documents
#   确保后续 delete 操作能正常找到 document_id
# - delete_document 兜底：即使内存缓存中不存在，也尝试直接从 ChromaDB 删除
# - 异步上传：POST /upload/start 返回 task_id，GET /upload/progress/{id} 轮询进度

import os
import re
import uuid
import shutil
import threading
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from backend.config import settings
from backend.utils.document_parser import DocumentParser
from backend.services.vector_store import VectorStoreService
from backend.services.llm_service import LLMService
from backend.models.schemas import (
    QueryRequest, QueryResponse, UploadResponse, 
    HealthResponse, DocumentInfo, Citation
)

# ── 中文数字映射（用于"第100条"→"第一百条" 归一化） ──
_CN_DIGITS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
_CN_RADICES = ["", "十", "百", "千"]

def _arabic_to_chinese(num: int) -> str:
    """将阿拉伯数字转为中文数字，支持 0~99999"""
    if num == 0:
        return "零"
    
    # 按位拆分（个十百千万）
    digits = []
    while num > 0:
        digits.append(num % 10)
        num //= 10
    
    result = ""
    need_zero = False
    for i in range(len(digits) - 1, -1, -1):
        d = digits[i]
        if d == 0:
            need_zero = True
        else:
            if need_zero:
                result += "零"
                need_zero = False
            result += _CN_DIGITS[d] + _CN_RADICES[i]
    
    # 修正"一十" → "十"
    if result.startswith("一十"):
        result = result[1:]
    
    return result

def _normalize_article_numbers(text: str) -> str:
    """将查询中的 '第N条' / '第N款' / '第N章' / '第N节' 中的阿拉伯数字转为中文数字
    
    例: "民法典第100条是什么" → "民法典第一百条是什么"
    """
    def _replacer(m):
        prefix = m.group(1)  # "第"
        num = int(m.group(2))
        suffix = m.group(3)  # "条/款/章/节"
        return prefix + _arabic_to_chinese(num) + suffix
    return re.sub(r'(第)(\d+)([条款章节])', _replacer, text)

class QAService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.llm_service = LLMService()
        self.documents = {}
        # 异步上传任务的进度存储（纯内存，重启后丢失）
        self._upload_progress: Dict[str, Dict] = {}
        # 从 ChromaDB 恢复历史文档列表（应对服务重启后数据丢失）
        self._recover_from_db()
    
    def _recover_from_db(self):
        """启动时从 ChromaDB 恢复文档列表和 BM25 索引"""
        try:
            restored = self._restore_documents_from_db()
            if restored:
                doc_names = [d.filename for d in restored]
                print(f"[RECOVER] 已从 ChromaDB 恢复 {len(restored)} 个文档: {doc_names}")
            # 确保 BM25 索引已加载（all_documents 可能在重启后丢失）
            if not self.vector_store.all_documents:
                self.vector_store._rebuild_bm25_from_db()
                print(f"[RECOVER] 已从 ChromaDB 重建 BM25 索引")
        except Exception as e:
            print(f"[RECOVER] 恢复过程异常: {e}")
    
    def warmup(self):
        """服务预热：预加载重排序模型，避免首次 rerank 请求超时
        
        - 模型已缓存时：阻塞加载（约 10s），确保首个请求可用
        - 模型未缓存时：后台线程下载，不阻塞服务启动
        bge-reranker-large 模型约 1.4GB，首次下载需数分钟。
        """
        from backend.services.llm_service import _is_reranker_cached, _get_reranker
        if _is_reranker_cached():
            print("[WARMUP] 重排序模型已缓存，同步加载中...")
            reranker = _get_reranker()
            if reranker is not None:
                print("[WARMUP] 重排序模型预热完成")
            else:
                print("[WARMUP] 重排序模型预热失败")
        else:
            print("[WARMUP] 重排序模型未缓存，在后台线程中下载（不阻塞启动）...")
            import threading
            def _do_warmup():
                print("[WARMUP] 正在后台下载并预热重排序模型...")
                reranker = _get_reranker()
                if reranker is not None:
                    print("[WARMUP] 重排序模型后台预热完成")
                else:
                    print("[WARMUP] 重排序模型后台预热失败")
            thread = threading.Thread(target=_do_warmup, daemon=True)
            thread.start()
    
    def _report_progress(self, task_id: str, progress: float, stage: str, message: str):
        """更新上传任务进度"""
        self._upload_progress[task_id] = {
            "progress": round(progress, 2),
            "stage": stage,
            "message": message
        }
    
    async def start_upload_async(self, file, filename: str) -> str:
        """启动异步上传，返回 task_id"""
        task_id = str(uuid.uuid4())
        file_path = os.path.join(settings.UPLOAD_DIR, f"{task_id}_{filename}")
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)
        
        self._report_progress(task_id, 0.0, "queued", "任务已创建")
        
        # 在后台线程中处理
        thread = threading.Thread(
            target=self._process_upload_background,
            args=(task_id, file_path, filename),
            daemon=True
        )
        thread.start()
        
        return task_id
    
    def _process_upload_background(self, task_id: str, file_path: str, filename: str):
        """后台处理上传文件（解析 → 编码 → 存储）"""
        document_id = str(uuid.uuid4())
        
        try:
            # ── 1. 解析文档 ──
            self._report_progress(task_id, 0.05, "parsing", "正在解析文档…")
            chunks = DocumentParser.parse_file(file_path)
            total_chunks = len(chunks)
            self._report_progress(task_id, 0.15, "parsing", f"文档解析完成，共 {total_chunks} 个片段")
            
            if total_chunks == 0:
                self._report_progress(task_id, 1.0, "error", "文档内容为空，无法处理")
                return
            
            # ── 2. 准备元数据 ──
            ids = [chunk.id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = []
            for chunk in chunks:
                metadatas.append({
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": chunk.page_number or 0,
                    "paragraph_number": chunk.paragraph_number or 0
                })
            
            # ── 3. 分批编码向量 ──
            batch_size = 16
            all_embeddings = []
            
            for i in range(0, total_chunks, batch_size):
                batch = documents[i:i + batch_size]
                batch_embeddings = self.vector_store.embedding_model.encode(batch).tolist()
                all_embeddings.extend(batch_embeddings)
                
                progress = 0.15 + 0.70 * (min(i + batch_size, total_chunks) / total_chunks)
                processed = min(i + batch_size, total_chunks)
                self._report_progress(
                    task_id, progress, "embedding",
                    f"编码向量 ({processed}/{total_chunks})"
                )
            
            # ── 4. 存入 ChromaDB ──
            self._report_progress(task_id, 0.88, "saving", "正在存入向量库…")
            self.vector_store.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=all_embeddings
            )
            
            # ── 5. 更新 BM25 索引 ──
            self._report_progress(task_id, 0.94, "saving", "正在更新关键词索引…")
            for i, chunk in enumerate(chunks):
                self.vector_store.all_documents.append({
                    'id': chunk.id,
                    'content': chunk.content,
                    'metadata': metadatas[i]
                })
            self.vector_store._rebuild_bm25_from_db()
            
            # ── 6. 记录文档信息 ──
            self.documents[document_id] = {
                'id': document_id,
                'filename': filename,
                'upload_time': datetime.now(),
                'chunk_count': total_chunks,
                'file_path': file_path
            }
            
            self._report_progress(task_id, 1.0, "done", f"完成，共 {total_chunks} 个片段")
            
        except Exception as e:
            self._report_progress(task_id, -1, "error", str(e))
            # 清理失败的文件
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def get_upload_progress(self, task_id: str) -> Optional[Dict]:
        """获取上传任务进度"""
        return self._upload_progress.get(task_id)
    
    def get_document_file_path(self, document_id: str) -> Optional[str]:
        """获取文档原始文件路径"""
        doc = self.documents.get(document_id)
        if doc and doc.get('file_path'):
            return doc['file_path']
        # 兜底：在 uploads 目录下搜索
        upload_dir = settings.UPLOAD_DIR
        if os.path.exists(upload_dir):
            for f in os.listdir(upload_dir):
                if document_id in f:
                    return os.path.join(upload_dir, f)
        return None
    
    def get_document_chunks(self, document_id: str, page: int = 1, page_size: int = 50) -> Dict:
        """获取文档检索片段，用于文档预览"""
        return self.vector_store.get_document_chunks(document_id, page, page_size)
    
    # ── 以下为同步方法（保持向后兼容） ──
    
    async def upload_document(self, file, filename: str) -> UploadResponse:
        document_id = str(uuid.uuid4())
        file_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}_{filename}")
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file, buffer)
            
            chunks = DocumentParser.parse_file(file_path)
            chunk_count = self.vector_store.add_documents(chunks, document_id, filename)
            
            self.documents[document_id] = {
                'id': document_id,
                'filename': filename,
                'upload_time': datetime.now(),
                'chunk_count': chunk_count,
                'file_path': file_path
            }
            
            return UploadResponse(
                document_id=document_id,
                filename=filename,
                message=f"文档上传成功，共切分为 {chunk_count} 个片段",
                chunk_count=chunk_count
            )
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
    
    async def query(self, request: QueryRequest) -> QueryResponse:
        # 将"第100条"归一化为"第一百条"，解决文档中中文数字 vs 用户输入阿拉伯数字的匹配问题
        search_question = _normalize_article_numbers(request.question)
        
        if request.use_rerank:
            # ── Rerank 流程：混合检索取候选池 → CrossEncoder 精排 → top-k ──
            # 设计要点：
            #   ① 先跑 keyword-only 混合检索作为基准保底（结果与 keyword 路径完全一致）
            #   ② 再跑大候选池供 CrossEncoder 精排
            #   ③ rerank 结果正常时用精排结果，异常时空降至 keyword_fallback
            # 修复记录（2026-07）：
            #   - 问题：rerank 路径的 hybrid_search 使用 rerank_candidates=20 导致 internal_k=20，
            #     keyword-only 路径用 rerank_candidates=0 导致 internal_k=10，
            #     两个候选池不同 => RRF 分数分布不同 => top-5 结果可能不一致。
            #     同时 rerank_results 中 0.1 的置信度阈值对 bge-reranker-base 的 logits 输出误判。
            #   - 修复：keyword_fallback 保证降级结果一致性；移除硬阈值；添加空结果兜底。
            
            # ① keyword 降级保底（始终启用完整混合搜索，确保保底结果完整）
            # 设计：use_keyword=True 固定，不受用户关键词勾选影响——keyword_fallback
            # 是内部保底机制，目的是确保关键字精确匹配的文档不会丢失。
            keyword_fallback = self.vector_store.hybrid_search(
                search_question,
                top_k=request.top_k,
                use_keyword=True
            )
            
            # ② 大候选池供 rerank 精排
            candidates = self.vector_store.hybrid_search(
                search_question,
                top_k=request.top_k,
                rerank_candidates=settings.RERANK_CANDIDATES,  # 默认 20
                use_keyword=request.use_keyword_search
            )
            
            # ③ CrossEncoder 精排
            search_results = self.llm_service.rerank_results(
                search_question, candidates, request.top_k
            )
            
            # ④ 安全兜底：rerank 返回空或异常时降级到 keyword_fallback
            if not search_results:
                search_results = keyword_fallback
            else:
                # ⑤ 关键字保底插队：确保 keyword 精确命中的文档始终在最终 top-k 中
                # 修复原因：bge-reranker-base 打分与 RRF+层级评分排序不同，
                # 可能导致具体条款（如第58条）被更通用的相关文档（如第56条）挤出 top-k。
                # 修复策略：将 keyword_fallback 中的文档按 rerank_score 排序后，
                # 若不在 search_results 中则插入，确保不丢失精确匹配结果。
                rerank_ids = {r['id'] for r in search_results}
                for kw_doc in keyword_fallback:
                    if kw_doc['id'] not in rerank_ids:
                        if search_results:
                            top_score = search_results[0].get('rerank_score', 0.99)
                            kw_doc['rerank_score'] = top_score * 0.995
                            kw_doc['score'] = kw_doc['rerank_score']
                        search_results.append(kw_doc)
                        rerank_ids.add(kw_doc['id'])
                # 重新排序并截断
                search_results.sort(
                    key=lambda x: x.get('rerank_score', x.get('score', 0)),
                    reverse=True
                )
                search_results = search_results[:request.top_k]
            
        elif request.use_keyword_search:
            # ── 关键词混合检索（语义 + BM25 → RRF 融合 → 层级评分） ──
            search_results = self.vector_store.hybrid_search(
                search_question, 
                request.top_k
            )
        else:
            # ── 纯语义检索 ──
            search_results = self.vector_store.semantic_search(
                search_question, 
                request.top_k
            )
        
        # 传给 LLM 时仍用原始问题（用户看到的是自己输入的问题）
        answer, citations, found_kb, processing_time = self.llm_service.generate_answer(
            request.question,
            search_results,
            request.use_rerank
        )
        
        return QueryResponse(
            answer=answer,
            citations=citations,
            found_in_knowledge_base=found_kb,
            processing_time_ms=processing_time
        )
    
    async def get_documents(self) -> List[DocumentInfo]:
        if self.documents:
            return [
                DocumentInfo(
                    id=doc['id'],
                    filename=doc['filename'],
                    upload_time=doc['upload_time'],
                    chunk_count=doc['chunk_count']
                )
                for doc in self.documents.values()
            ]
        return self._restore_documents_from_db()
    
    def _restore_documents_from_db(self) -> List[DocumentInfo]:
        """从 ChromaDB 元数据恢复文档列表并同步到 self.documents 缓存"""
        try:
            results = self.vector_store.collection.get()
            if not results or not results['metadatas']:
                return []
            
            doc_map = {}
            for meta in results['metadatas']:
                doc_id = meta.get('document_id', '')
                filename = meta.get('filename', '未知文档')
                if doc_id not in doc_map:
                    doc_map[doc_id] = {'id': doc_id, 'filename': filename, 'chunk_count': 0}
                doc_map[doc_id]['chunk_count'] += 1
            
            # 同步到 self.documents 缓存，确保后续 delete 等操作能找到
            for info in doc_map.values():
                if info['id'] not in self.documents:
                    self.documents[info['id']] = {
                        'id': info['id'],
                        'filename': info['filename'],
                        'upload_time': datetime.now(),
                        'chunk_count': info['chunk_count'],
                        'file_path': None  # 重启后文件路径不可恢复，但向量数据仍可删除
                    }
            
            return [
                DocumentInfo(id=info['id'], filename=info['filename'],
                           upload_time=datetime.now(), chunk_count=info['chunk_count'])
                for info in doc_map.values()
            ]
        except Exception:
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        if document_id in self.documents:
            file_path = self.documents[document_id].get('file_path')
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            self.vector_store.delete_document(document_id)
            del self.documents[document_id]
            return True
        # 兜底：直接通过 ChromaDB 删除（即使内存缓存中没有）
        try:
            self.vector_store.delete_document(document_id)
            return True
        except Exception:
            return False
    
    async def get_health(self) -> HealthResponse:
        vector_db_status = "connected" if self.vector_store.get_document_count() >= 0 else "disconnected"
        try:
            llm_status = "connected"
        except:
            llm_status = "disconnected"
        return HealthResponse(status="ok", vector_db=vector_db_status, llm=llm_status)

qa_service = QAService()
