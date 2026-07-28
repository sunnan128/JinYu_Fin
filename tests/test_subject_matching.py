# 主体关键词模糊匹配回归测试
# 背景：用户提问"中华人民共和国国家赔偿法（２０１２年）的第十一条是什么"时，
# 截取的 subject_key 含全角括号/数字 + 末尾"的"，与库内半角文件名直接 `in`
# 匹配失败，导致主体补充检索落空、该法规条文检索不到。修复后用核心法律名
# 双向包含（含全角→半角归一、去 UUID/扩展名/年份括号/助词）。
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.vector_store import (
    VectorStoreService,
    _subject_matches,
    _core_law_name,
)


PDF_FN = "a7b4ffe3-e7a4-4965-a9c4-b12ddbcd9780_中华人民共和国国家赔偿法（2012）.pdf"
MD_FN = "830c64bf-b51b-45ab-8372-39a53c628507_009_中华人民共和国信托法.md"


def test_core_law_name_normalizes():
    # 全角年份+末尾"的" → 核心法律名
    assert _core_law_name("中华人民共和国国家赔偿法（２０１２年）的") == "中华人民共和国国家赔偿法"
    # 带 UUID 前缀 + 扩展名 + 半角年份 → 核心法律名
    assert _core_law_name(PDF_FN) == "中华人民共和国国家赔偿法"
    assert _core_law_name(MD_FN) == "中华人民共和国信托法"


def test_subject_matches_fullwidth_vs_halfwidth():
    subj = "中华人民共和国国家赔偿法（２０１２年）的"
    assert _subject_matches(subj, PDF_FN) is True


def test_subject_matches_abbreviation():
    # 用户只说简称也能匹配到全称文件名
    assert _subject_matches("信托法", MD_FN) is True
    assert _subject_matches("赔偿法", PDF_FN) is True


def test_subject_matches_negative():
    # 不同法律不应误匹配
    assert _subject_matches("证券法", MD_FN) is False
    assert _subject_matches("信托法", PDF_FN) is False


def test_search_by_subject_metadata_recalls_pdf():
    # 模拟"国家赔偿法 PDF + 信托法 md"共存，主体检索应只召回国家赔偿法
    vs = object.__new__(VectorStoreService)
    vs.all_documents = [
        {"id": 0, "content": "第十一条 赔偿请求人…", "metadata": {"filename": PDF_FN}},
        {"id": 1, "content": "第二十二条 受托人违反…", "metadata": {"filename": MD_FN}},
    ]
    subj, _ = VectorStoreService._parse_hierarchical_query(
        "国家赔偿法（２０１２年）的第十一条是什么"
    )
    hits = vs._search_by_subject_metadata(subj)
    assert len(hits) == 1
    assert hits[0]["id"] == 0


def test_apply_hierarchical_scoring_boosts_target_clause():
    # 修复后：国家赔偿法第十一条应被加分并排序到最前，不被同名其他法律挤掉
    subj, sub = VectorStoreService._parse_hierarchical_query(
        "国家赔偿法（２０１２年）的第十一条是什么"
    )
    scored = [
        {"id": 0, "content": "第十一条 赔偿请求人有权向…", "metadata": {"filename": PDF_FN}, "score": 0.0},
        {"id": 1, "content": "第二十二条 其他法律条款", "metadata": {"filename": MD_FN}, "score": 0.0},
    ]
    VectorStoreService._apply_hierarchical_scoring(scored, subj, sub)
    # 目标条款应获 FULL_MATCH_BOOST（0.03），未命中主体的同名条款不加
    assert next(it for it in scored if it["id"] == 0)["score"] == 0.03
    assert next(it for it in scored if it["id"] == 1)["score"] == 0.0
    scored.sort(key=lambda x: x["score"], reverse=True)
    assert scored[0]["id"] == 0
