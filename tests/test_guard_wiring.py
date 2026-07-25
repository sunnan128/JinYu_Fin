# -*- coding: utf-8 -*-
"""三层幻觉抑制接线集成测试。

目标：验证 qa_service.query() 真正接入 L1/L2/L3，且
  上传 / 解析 / 检索 / 拦截-打标-重答 全链路无异常。
不依赖真实 embedding 模型与 LLM API（用假对象替换底层服务）。
"""
import os
import sys
import asyncio
import tempfile

import pytest


# ───────────────────────── 假底层服务 ─────────────────────────
class FakeCollection:
    def __init__(self):
        self.store = {}

    def add(self, ids, documents, metadatas, embeddings):
        for i, _id in enumerate(ids):
            self.store[_id] = (documents[i], metadatas[i])

    def get(self, where=None):
        return {
            "ids": list(self.store.keys()),
            "documents": [v[0] for v in self.store.values()],
            "metadatas": [v[1] for v in self.store.values()],
        }

    def query(self, query_embeddings, n_results):
        ids = list(self.store.keys())[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.store[i][0] for i in ids]],
            "metadatas": [[self.store[i][1] for i in ids]],
            "distances": [[0.1] * len(ids)],
        }

    def count(self):
        return len(self.store)

    def delete(self, ids):
        for i in ids:
            self.store.pop(i, None)


class FakeVectorStore:
    def __init__(self):
        self.collection = FakeCollection()
        self.all_documents = []
        self._last_added_metadatas = []
        self.candidates = []  # 由测试注入：模拟检索返回的候选

    def add_documents(self, chunks, document_id, filename):
        metas = []
        for c in chunks:
            m = {
                "document_id": document_id,
                "filename": filename,
                "page_number": c.page_number or 0,
                "paragraph_number": c.paragraph_number or 0,
            }
            m.update(c.metadata)  # 与真实 add_documents 一致：携带 front-matter
            metas.append(m)
            self.all_documents.append({"id": c.id, "content": c.content, "metadata": m})
        self._last_added_metadatas = metas
        return len(chunks)

    def hybrid_search(self, query, top_k=5, rerank_candidates=0, use_keyword=True):
        return list(self.candidates)

    def semantic_search(self, query, top_k=5):
        return list(self.candidates)

    def keyword_search(self, query, top_k=5):
        return list(self.candidates)

    def get_document_count(self):
        return len(self.all_documents)

    def delete_document(self, document_id):
        return 0

    def _rebuild_bm25_from_db(self):
        pass

    def _load_bm25_from_disk(self):
        pass

    def get_document_chunks(self, document_id, page=1, page_size=50):
        return {"total": 0, "chunks": []}


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.first_answer_abolished = True  # 模拟模型第一次生成时引用已废止条款

    def rerank_results(self, question, candidates, top_k=5):
        return candidates[:top_k]

    def generate_answer(self, question, search_results, use_rerank=True):
        self.calls.append({"question": question, "search_results": search_results})
        if len(self.calls) == 1 and self.first_answer_abolished:
            # 第一次：返回引用已废止条款（存贷比75%）的幻觉答案
            return ("根据《商业银行法》第三十九条，商业银行存贷比不得超过75%。", [], True, 12.0)
        # 重答或干净场景：返回现行有效内容
        return ("根据现行有效的《商业银行流动性风险管理办法》，商业银行应满足流动性覆盖率等监管指标。",
                [], True, 12.0)


# ───────────────────────── 夹具：替换底层后导入 qa_service ─────────────────────────
@pytest.fixture
def qa(monkeypatch):
    import backend.services.vector_store as vs_mod
    import backend.services.llm_service as llm_mod
    monkeypatch.setattr(vs_mod, "VectorStoreService", FakeVectorStore)
    monkeypatch.setattr(llm_mod, "LLMService", FakeLLM)
    import backend.services.qa_service as qs_mod
    svc = qs_mod.QAService()
    return svc, qs_mod


# ───────────────────────── 测试：上传 + 解析 + 元数据落库 ─────────────────────────
def test_upload_parse_stores_front_matter(qa):
    svc, _ = qa
    md = (
        "# 权威机构: 中国人民银行\n"
        "# 令号: 国务院令第768号\n"
        "# 排序键: 768\n"
        "# 施行日期: 2025-01-01\n"
        "# 效力状态: current\n\n"
        "第一条 为规范XX，制定本办法。\n"
        "第二条 本办法自发布之日起施行。\n"
    )
    p = os.path.join(tempfile.gettempdir(), "tmp_wiring_upload.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(md)
    resp = asyncio.run(svc.upload_document(open(p, "rb"), "测试办法.md"))
    os.remove(p)

    metas = svc.vector_store._last_added_metadatas
    assert metas, "应有入库元数据"
    assert any(m.get("权威机构") == "中国人民银行" for m in metas)
    assert any(m.get("效力状态") == "current" for m in metas)
    assert resp.chunk_count > 0


# ───────────────────────── 测试：检索正常 + 答案干净时不拦截 ─────────────────────────
def test_query_allows_clean_answer(qa):
    svc, qs_mod = qa
    svc.llm_service.first_answer_abolished = False  # 让模型直接给干净答案
    svc.vector_store.candidates = [
        {"id": "c2", "content": "流动性覆盖率监管要求。",
         "metadata": {"filename": "流动性办法.md", "效力状态": "current",
                      "权威机构": "金融监管总局", "令号": "流动性办法", "排序键": "2018"},
         "score": 0.8},
    ]
    from backend.models.schemas import QueryRequest
    result = asyncio.run(svc.query(QueryRequest(question="流动性监管指标有哪些")))
    assert result.guard is not None
    assert result.guard.blocked is False
    assert result.guard.action == "allow"
    assert len(svc.llm_service.calls) == 1  # 未触发重答
    assert "流动性" in result.answer


# ───────────────────────── 测试：检索召回含已废止块时 L1 拦截 ─────────────────────────
def test_l1_filters_abolished_chunk_before_llm(qa):
    svc, qs_mod = qa
    svc.llm_service.first_answer_abolished = False
    svc.vector_store.candidates = [
        {"id": "c1", "content": "商业银行法第三十九条 存贷比不得超过75%。",
         "metadata": {"filename": "商业银行法.md", "效力状态": "已废止",
                      "权威机构": "全国人大", "令号": "商业银行法", "排序键": "1995"},
         "score": 0.9},
        {"id": "c2", "content": "流动性覆盖率监管要求。",
         "metadata": {"filename": "流动性办法.md", "效力状态": "current",
                      "权威机构": "金融监管总局", "令号": "流动性办法", "排序键": "2018"},
         "score": 0.8},
    ]
    from backend.models.schemas import QueryRequest
    asyncio.run(svc.query(QueryRequest(question="商业银行存贷比规定")))
    # 喂给 LLM 的检索结果不应含已废止块
    first_call = svc.llm_service.calls[0]
    for r in first_call["search_results"]:
        assert r["metadata"].get("效力状态") != "已废止"


# ───────────────────────── 测试：L3 拦截 + 打标 + 要求重答 ─────────────────────────
def test_query_blocks_tags_and_regenerates_on_abolished_reference(qa):
    svc, qs_mod = qa
    svc.vector_store.candidates = [
        {"id": "c2", "content": "流动性覆盖率监管要求。",
         "metadata": {"filename": "流动性办法.md", "效力状态": "current",
                      "权威机构": "金融监管总局", "令号": "流动性办法", "排序键": "2018"},
         "score": 0.8},
    ]
    from backend.models.schemas import QueryRequest
    result = asyncio.run(svc.query(QueryRequest(question="商业银行存贷比规定是什么")))

    # L3 拦截 + 打标
    assert result.guard is not None
    assert result.guard.blocked is True
    assert result.guard.action == "regenerate"
    assert any(h["entry_id"] == "commercial_bank_ltv_75" for h in result.guard.hits)

    # 要求重答：LLM 被调用 >= 2 次
    assert len(svc.llm_service.calls) >= 2

    # 最终答案已替换为干净版本（不再含已废止条款内容）
    assert "存贷比不得超过75%" not in result.answer
    assert "流动性" in result.answer


# ───────────────────────── 测试：整体可正常导入运行 ─────────────────────────
def test_imports_ok(monkeypatch):
    import backend.services.vector_store as vs_mod
    import backend.services.llm_service as llm_mod
    monkeypatch.setattr(vs_mod, "VectorStoreService", FakeVectorStore)
    monkeypatch.setattr(llm_mod, "LLMService", FakeLLM)
    import backend.main  # 导入不应抛异常（验证整体可正常运行）
    import backend.services.qa_service as qs_mod
    assert qs_mod is not None
