# -*- coding: utf-8 -*-
"""ingest_raw.py 测试：解析保留元数据 / 灌库携带元数据 / L1 消费 / 去重。

目标：验证 Phase D 数据底座衔接后，L1（检索端权威度过滤）不再空转——
入库的 chunk 真正携带 权威机构/效力状态 等 front-matter 元数据。
全程用 FakeVectorStore 替换真实 ChromaDB/嵌入模型，不污染真实库，也不依赖网络。
"""
import os
import sys
import asyncio
import tempfile

import pytest

# 确保 backend 包可导入
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ───────────────────────── 假底层服务（与 test_guard_wiring 同构） ─────────────────────────
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


@pytest.fixture
def qa(monkeypatch):
    import backend.services.vector_store as vs_mod
    monkeypatch.setattr(vs_mod, "VectorStoreService", FakeVectorStore)
    import backend.services.qa_service as qs_mod
    svc = qs_mod.QAService()
    return svc


# ───────────────────────── 1. 解析真实 raw 文件保留元数据 ─────────────────────────
def test_collect_raw_preserves_front_matter_metadata():
    """collect_raw_documents 解析出的 chunk 必须携带 权威机构/效力状态，
    否则 L1 在真实链路仍会空转。"""
    import ingest_raw
    from backend.utils.document_parser import DocumentParser

    raw_md = (
        "# 标题: 测试办法\n"
        "# 来源: https://example.com/x\n"
        "# 权威机构: central_bank\n"
        "# 令号: 国务院令第768号\n"
        "# 施行日期: 2025-01-01\n"
        "# 效力状态: current\n"
        "# 排序键: 768\n\n"
        "第一条 为规范XX，制定本办法。\n"
        "第二条 本办法自发布之日起施行。\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "001_测试办法.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(raw_md)
        docs = ingest_raw.collect_raw_documents(td)
        assert docs, "应解析出文档"
        assert len(docs) == 1
        filename, chunks = docs[0]
        assert filename == "001_测试办法.md"
        # 每个 chunk 都应带 front-matter 元数据
        for c in chunks:
            assert c.metadata.get("权威机构") == "central_bank"
            assert c.metadata.get("效力状态") == "current"
            # 正文不应包含元数据噪声行
            assert "# 权威机构:" not in c.content


# ───────────────────────── 2. 灌库把元数据带入 store（复用上传路径） ─────────────────────────
def test_ingest_carries_metadata_into_store(qa):
    """ingest_documents 复用 upload_document 路径，元数据应进入向量库。"""
    import ingest_raw

    raw_md = (
        "# 权威机构: 中国人民银行\n"
        "# 令号: 国务院令第768号\n"
        "# 排序键: 768\n"
        "# 施行日期: 2025-01-01\n"
        "# 效力状态: current\n\n"
        "第一条 为规范XX，制定本办法。\n"
        "第二条 本办法自发布之日起施行。\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "001_测试办法.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(raw_md)
        n = ingest_raw.ingest_documents(qa, td, force=True, dry_run=False)

    assert n > 0
    # 元数据确实进入了 store
    all_meta = [d["metadata"] for d in qa.vector_store.all_documents]
    assert any(m.get("权威机构") == "中国人民银行" for m in all_meta)
    assert any(m.get("效力状态") == "current" for m in all_meta)
    # QAService 文档注册表也记录了该文件
    assert "001_测试办法.md" in {d.filename for d in asyncio.run(qa.get_documents())}


# ───────────────────────── 3. A 让 L1 能消费：abolished 块被剔除 ─────────────────────────
def test_ingest_raw_enables_l1_to_drop_abolished():
    """合成一份 效力状态: abolished 的 raw 文档，经解析后交给 L1 过滤，
    证明灌库带来的元数据能让 L1 真正拦截（这正是空转 vs 不空转 的区别）。"""
    import ingest_raw
    from backend.utils.document_parser import DocumentParser
    from backend.utils.hallucination_guard import filter_by_authority

    abolished_md = (
        "# 权威机构: 国务院\n"
        "# 令号: 旧规定第1号\n"
        "# 排序键: 1\n"
        "# 施行日期: 2000-01-01\n"
        "# 效力状态: abolished\n\n"
        "第一条 旧规规定存款利率不得超过某上限。\n"
    )
    current_md = (
        "# 权威机构: 中国人民银行\n"
        "# 令号: 新规第2号\n"
        "# 排序键: 2\n"
        "# 施行日期: 2025-01-01\n"
        "# 效力状态: current\n\n"
        "第一条 新规规定利率由市场决定。\n"
    )
    with tempfile.TemporaryDirectory() as td:
        pa = os.path.join(td, "old.md")
        pb = os.path.join(td, "new.md")
        with open(pa, "w", encoding="utf-8") as f:
            f.write(abolished_md)
        with open(pb, "w", encoding="utf-8") as f:
            f.write(current_md)
        docs = ingest_raw.collect_raw_documents(td)

    # 把所有 chunk 展平，模拟"检索召回候选"，交给 L1
    candidates = []
    for _fn, chunks in docs:
        for c in chunks:
            candidates.append(c)
    result = filter_by_authority(candidates)

    # 已废止块被剔除
    assert len(result.removed_abolished) == 1
    assert result.removed_abolished[0].metadata.get("效力状态") == "abolished"
    # 现行有效块保留
    assert len(result.kept) == 1
    assert result.kept[0].metadata.get("效力状态") == "current"


# ───────────────────────── 4. 去重：已存在文件被跳过 ─────────────────────────
def test_ingest_skips_existing_documents(qa):
    """库里已有同名文件时，默认去重跳过，不重复灌。"""
    import ingest_raw

    raw_md = (
        "# 权威机构: 中国人民银行\n"
        "# 效力状态: current\n\n"
        "第一条 测试内容。\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "001_重复测试.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(raw_md)
        # 先灌一次
        n1 = ingest_raw.ingest_documents(qa, td, force=True, dry_run=False)
        assert n1 > 0
        # 再灌一次（默认去重）
        n2 = ingest_raw.ingest_documents(qa, td, force=False, dry_run=False)
        assert n2 == 0, "去重应跳过已存在文件"
        # --force 可强制重灌
        n3 = ingest_raw.ingest_documents(qa, td, force=True, dry_run=False)
        assert n3 > 0
