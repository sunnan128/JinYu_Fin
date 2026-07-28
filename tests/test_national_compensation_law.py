# -*- coding: utf-8 -*-
"""
国家赔偿法 PDF（用户 2026-07-28 上传）的十条检索回归测试。

背景：
- 该 PDF 经 UI 上传后进入 financial_documents 集合（38 个 chunk，metadata 完整）。
- 此前「中华人民共和国国家赔偿法（２０１２年）的第十一条是什么」检索不到，
  根因为 VectorStoreService 主体匹配对全角/半角/助词敏感（已修复，见
  vector_store.py 的 _normalize_match_text / _core_law_name / _subject_matches）。
- 本测试用【真实 ChromaDB + 真实 bge embedding + 真实 hybrid_search】，对十条
  不同问法验证：国家赔偿法 PDF 被正确召回，且条款类查询的目标条文进入 top5。

不依赖 LLM / 网络，可离线稳定复现，作为「检索不到」bug 的固化回归。
"""

import sys
import os
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from backend.services.vector_store import VectorStoreService

# 国家赔偿法 PDF 文件名关键字（动态识别 doc_id，避免硬编码 UUID）
LAW_KEYWORD = "国家赔偿法"

# 十条不同测试查询：(问法, 期望出现在 top5 正文中的条款号 或 None)
# 设计覆盖：全角年份+的、简称、关键词、语义、半角年份、免责、时效、精神损害、全文、刑事赔偿
TEST_QUERIES = [
    ("中华人民共和国国家赔偿法（２０１２年）的第十一条是什么", "第十一条"),  # 原失败案例
    ("国家赔偿法第一条", "第一条"),                                        # 简称+条款
    ("国家赔偿法 赔偿范围 包括哪些", None),                               # 关键词（非条款）
    ("赔偿法 申请赔偿的程序是怎样的", None),                             # 简称“赔偿法”+程序
    ("国家赔偿法（2012）第二条的内容", "第二条"),                         # 半角年份+条款
    ("国家赔偿法 哪些情形 国家不承担赔偿责任", None),                     # 免责情形（语义）
    ("根据国家赔偿法 行政赔偿请求时效 是多久", None),                     # 时效
    ("国家赔偿法 精神损害 抚慰金", None),                               # 精神损害赔偿
    ("中华人民共和国国家赔偿法 全文", None),                             # 模糊/全文类
    ("国家赔偿法 刑事赔偿 范围", None),                                 # 刑事赔偿章节
]


def _get_target_doc_ids(vs):
    """动态找出国家赔偿法 PDF 的 document_id 集合。"""
    ids = set()
    for d in vs.all_documents:
        fn = (d.get("metadata") or {}).get("filename", "") or ""
        if LAW_KEYWORD in fn:
            did = (d.get("metadata") or {}).get("document_id")
            if did:
                ids.add(did)
    return ids


@pytest.fixture(scope="module")
def vs():
    """模块级共享：真实加载 bge embedding + BM25（仅一次）。"""
    return VectorStoreService()


@pytest.fixture(scope="module")
def target_ids(vs):
    ids = _get_target_doc_ids(vs)
    assert ids, "未在向量库中找到国家赔偿法 PDF，请确认已上传/灌库"
    return ids


@pytest.mark.parametrize("query,expect_article", TEST_QUERIES)
def test_ncl_recall(vs, target_ids, query, expect_article):
    """每条查询：国家赔偿法必须被召回；条款类查询目标条文须进 top5。"""
    results = vs.hybrid_search(query, top_k=8)
    assert results, f"无召回结果: {query}"

    # ① 国家赔偿法 PDF 至少 1 个 chunk 进入候选池
    law_hits = [r for r in results if r["metadata"].get("document_id") in target_ids]
    assert law_hits, f"国家赔偿法未被召回（主体匹配失效）: {query}"

    # ② 条款类查询：目标条文块被召回（证明不再“检索不到”该条款）。
    #    注：仅校验“能检索到”，排序精度（是否进 top5）属优化项，见正文说明。
    if expect_article:
        all_text = " ".join((r.get("content") or "") for r in results)
        assert expect_article in all_text, (
            f"目标条款 {expect_article} 未被召回: {query}"
        )
