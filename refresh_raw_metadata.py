# -*- coding: utf-8 -*-
"""一次性刷新：删除库中已存在的 17 部 raw（无论是否带元数据），再带元数据重新灌入。

背景：用户此前经 UI 上传的 15 部 raw .md 是在"元数据入库修复"之前上传的，
导致 chunk 缺 权威机构/效力状态 字段，L1 对这批数据空转。本脚本把 17 部
按 filename 删除后重新灌入，使其全部携带 front-matter 元数据，让 L1 真正生效。
仅操作 17 部 raw（按爬虫命名 001_/002_/...），不触碰 sample_test_docs 等其它文档。
"""
import os
import sys
import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RAW_DIR = r"D:\workbuddy_pro\finance_rag_data\raw"
CHROMA_DIR = "backend/data/chroma"


def main():
    import chromadb
    from backend.services.qa_service import QAService
    from ingest_raw import ingest_documents

    names = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(RAW_DIR, "*.md")))]
    print(f"[INFO] 待刷新文件数：{len(names)}")

    # 1. 删除库中已存在的这 17 部（按 filename，覆盖重复副本）
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection("financial_documents")
    total_deleted = 0
    for name in names:
        r = col.get(where={"filename": name})
        if r and r.get("ids"):
            col.delete(ids=r["ids"])
            total_deleted += len(r["ids"])
            print(f"  [DELETE] {name}: {len(r['ids'])} 块")
    print(f"[INFO] 共删除 {total_deleted} 块（旧副本）")

    # 2. 重新灌入全部 17 部（此时库里已无，dedup 全新增，带元数据）
    qa = QAService()
    ingest_documents(qa, RAW_DIR, force=True, dry_run=False)


if __name__ == "__main__":
    main()
