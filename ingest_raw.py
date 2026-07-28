# -*- coding: utf-8 -*-
"""批量灌库脚本：把带 front-matter 元数据的法规 .md 灌入向量库（Phase D 数据底座衔接）。

用途
====
金语 AI 的三层幻觉抑制中，L1（检索端权威度过滤）依赖每个 chunk 的元数据
（权威机构 / 效力状态 / 施行日期 / 排序键 / 令号）。但用户前端上传的 PDF/Word
往往不带这些字段，导致 L1 在真实链路空转。

本脚本把 `finance_rag_data/raw/*.md`（爬虫产线已写入 front-matter）批量灌库，
使向量库里的 chunk 真正携带时效元数据，从而让 L1 不再空转、可依据权威度/版本排序。

设计约束（仅新增、不改任何现有功能）
====================================
- 复用既有链路：DocumentParser.parse_file 解析 + QAService.upload_document 入库，
  与前端上传走完全相同的入库路径（meta.update(chunk.metadata) 携带元数据）。
- 不新增/修改任何 service / parser / schema 代码。
- 按文件名去重：已存在于库中的文档默认跳过（--force 可强制重灌）。
- 单文件失败不影响整体：每个文件独立 try/except。

运行
====
    # 默认来源 D:/workbuddy_pro/finance_rag_data/raw，去重灌库
    python ingest_raw.py

    # 指定来源目录
    python ingest_raw.py --source "D:/path/to/raw"

    # 强制重灌（忽略已存在）
    python ingest_raw.py --force

    # 仅预览会灌哪些文件、多少块，不写库
    python ingest_raw.py --dry-run
"""
import os
import sys
import glob
import asyncio
import argparse


# ── 项目根目录（确保 backend 包可导入） ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 默认来源：爬虫产线输出目录（finance_rag_data/crawl_regulations.py 落盘处）
DEFAULT_RAW_DIR = r"D:\workbuddy_pro\finance_rag_data\raw"


def collect_raw_documents(source_dir: str):
    """【纯函数·可单测】扫描 source_dir 下所有 .md，逐文件解析为 (filename, chunks)。

    不接触向量库/嵌入模型，仅依赖 DocumentParser。返回 list[(filename, List[DocumentChunk])]。
    解析失败的文件打印告警并跳过，不影响其余文件。
    """
    from backend.utils.document_parser import DocumentParser

    results = []
    md_paths = sorted(glob.glob(os.path.join(source_dir, "*.md")))
    if not md_paths:
        return results
    for path in md_paths:
        filename = os.path.basename(path)
        try:
            chunks = DocumentParser.parse_file(path)
        except Exception as e:  # 单文件失败不阻断整体
            print(f"  [WARN] 解析失败，已跳过：{filename} -> {e}")
            continue
        if not chunks:
            print(f"  [WARN] 解析为空，已跳过：{filename}")
            continue
        results.append((filename, chunks))
    return results


def _existing_filenames(qa_service) -> set:
    """读取库中已有文档文件名集合，用于去重。"""
    try:
        docs = asyncio.run(qa_service.get_documents())
        return {d.filename for d in docs}
    except Exception:
        return set()


def ingest_documents(qa_service, source_dir: str, force: bool = False,
                     dry_run: bool = False) -> int:
    """把 source_dir 下的 .md 灌入库。

    返回本次实际新增的 chunk 总数（dry_run 下返回将新增的预估总数）。
    """
    docs = collect_raw_documents(source_dir)
    if not docs:
        print(f"[INFO] 在 {source_dir} 未找到可入库的 .md 文件")
        return 0

    already = set() if force else _existing_filenames(qa_service)
    total_chunks = 0
    ingested_files = 0
    skipped_files = 0

    for filename, chunks in docs:
        if filename in already:
            print(f"  [SKIP] 已存在，跳过：{filename}")
            skipped_files += 1
            continue
        if dry_run:
            print(f"  [DRY] 将入库：{filename}  ({len(chunks)} 块)")
            total_chunks += len(chunks)
            ingested_files += 1
            continue
        # 真实灌库：复用与前端上传完全一致的入库路径
        file_path = os.path.join(source_dir, filename)
        try:
            with open(file_path, "rb") as f:
                resp = asyncio.run(qa_service.upload_document(f, filename))
            total_chunks += resp.chunk_count
            ingested_files += 1
            print(f"  [OK]   {filename}  ({resp.chunk_count} 块)")
        except Exception as e:
            print(f"  [ERROR] 入库失败，已跳过：{filename} -> {e}")

    action = "预览(DRY-RUN)" if dry_run else "入库"
    print(
        f"\n[SUMMARY] {action}完成：成功 {ingested_files} 个文件 / "
        f"跳过 {skipped_files} 个已存在 / 共 {total_chunks} 个片段"
    )
    return total_chunks


def main():
    parser = argparse.ArgumentParser(
        description="把带 front-matter 的法规 .md 批量灌入向量库（Phase D 数据底座衔接）"
    )
    parser.add_argument(
        "--source", default=DEFAULT_RAW_DIR,
        help=f"来源目录（默认：{DEFAULT_RAW_DIR}）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重灌，忽略已存在文档（默认按文件名去重跳过）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览将灌哪些文件/多少块，不写库",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"[ERROR] 来源目录不存在：{args.source}")
        sys.exit(1)

    # DRY-RUN 预览：仅解析、不构造 QAService，避免加载嵌入模型/ChromaDB
    if args.dry_run:
        print(f"[INFO] 来源目录：{args.source}")
        print(f"[INFO] 模式：DRY-RUN 预览（不写库）")
        docs = collect_raw_documents(args.source)
        total = sum(len(chunks) for _, chunks in docs)
        print(f"\n[INFO] 发现 {len(docs)} 个 .md 文件，共 {total} 个片段")
        for filename, chunks in docs:
            print(f"  [DRY] 将入库：{filename}  ({len(chunks)} 块)")
        print(
            f"\n[SUMMARY] 预览完成：{len(docs)} 个文件 / 共 {total} 个片段"
            f"（去重与写库请在非 dry-run 模式下执行）"
        )
        return

    print(f"[INFO] 来源目录：{args.source}")
    print(f"[INFO] 模式：{'强制重灌' if args.force else '去重灌库'}")

    # 延迟导入：仅在真正运行时才构造 QAService（会加载嵌入模型/ChromaDB）
    from backend.services.qa_service import QAService
    qa_service = QAService()

    ingest_documents(qa_service, args.source, force=args.force, dry_run=False)


if __name__ == "__main__":
    main()
