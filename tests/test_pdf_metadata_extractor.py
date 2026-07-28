# -*- coding: utf-8 -*-
"""pdf_metadata_extractor 单测 + 集成测试。

覆盖：
  - 单字段抽取（权威机构 / 令号 / 施行日期 / 效力状态）
  - 与 .md front-matter 同构的键完整性
  - 用 PyMuPDF 生成真实 PDF 的端到端抽取（含异常兜底）
"""
import os
import sys
import tempfile

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.utils.pdf_metadata_extractor import (
    extract_metadata_from_text,
    extract_pdf_metadata,
    _detect_authority,
    _extract_order,
    _extract_effective_date,
    _extract_status,
)


# ---------- 单字段 ----------
def test_detect_authority():
    assert _detect_authority("中国人民银行令〔2024〕第1号") == "central_bank"
    assert _detect_authority("中国证券监督管理委员会公告") == "csrc"
    assert _detect_authority("国家金融监督管理总局规定") == "nfra"
    assert _detect_authority("一段无关文字") == "other"


def test_extract_order_state_council():
    order, key = _extract_order("根据《国务院令第768号》的规定")
    assert order == "令号第768号"
    assert key == 768


def test_extract_order_pbc_year_num():
    order, key = _extract_order("中国人民银行令〔2024〕第1号")
    assert "2024" in order and "1号" in order
    assert key == 2024 * 1000 + 1


def test_extract_order_none():
    order, key = _extract_order("没有任何令号的正文")
    assert order == ""
    assert key is None


def test_extract_effective_date():
    assert _extract_effective_date("自2024年5月1日起施行") == "2024-05-01"
    assert _extract_effective_date("（2007年1月1日施行）") in ("2007-01-01",)
    assert _extract_effective_date("无日期") == ""


def test_extract_status_abolished_self():
    assert _extract_status("本办法已被废止") == "abolished"
    assert _extract_status("本决定予以废止") == "abolished"


def test_extract_status_current():
    assert _extract_status("现予公布，自2024年5月1日起施行") == "current"


def test_extract_status_unknown():
    # 不含"同时废止某某"这类外部废止，也不含现予公布/自X起施行 → unknown
    assert _extract_status("本法规与某旧办法同时废止其他规定") == "unknown"
    assert _extract_status("一段无关文字") == "unknown"


# ---------- 同构键完整性 ----------
def test_metadata_keys_match_front_matter():
    meta = extract_metadata_from_text(
        "中国人民银行令〔2024〕第1号\n现予公布，自2024年5月1日起施行",
        title="某管理办法",
    )
    for k in ["标题", "来源", "权威机构", "令号", "施行日期", "效力状态", "排序键"]:
        assert k in meta, f"缺少键: {k}"
    assert meta["标题"] == "某管理办法"
    assert meta["权威机构"] == "central_bank"
    assert meta["施行日期"] == "2024-05-01"
    assert meta["效力状态"] == "current"
    assert meta["排序键"] == 2024 * 1000 + 1
    # 透明标注
    assert meta["_meta_source"] == "pdf_heuristic"
    assert meta["_meta_confidence"] in ("medium", "high")


def test_metadata_unknown_is_safe_default():
    meta = extract_metadata_from_text("一段没有元信息的普通文本")
    assert meta["效力状态"] == "unknown"
    assert meta["权威机构"] == "other"
    assert meta["_meta_confidence"] == "low"


# ---------- 集成：用 PyMuPDF 生成真实 PDF ----------
def _make_pdf(path: str, text: str):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    # 用内置简体中文字体 china-s，确保中文可被 get_text() 抽取
    # （默认西文字体无中文字形，抽不出中文，会误伤测试；真实法规 PDF 自带 CJK 字体不受影响）
    try:
        page.insert_text((50, 60), "中华人民共和国某管理办法", fontname="china-s")
        page.insert_text((50, 90), text, fontname="china-s")
    except Exception:
        page.insert_text((50, 60), "中华人民共和国某管理办法")
        page.insert_text((50, 90), text)
    doc.save(path)
    doc.close()


def test_extract_pdf_metadata_integration():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "test.pdf")
    _make_pdf(p, "中国人民银行令〔2024〕第1号\n现予公布，自2024年5月1日起施行")
    meta = extract_pdf_metadata(p)
    assert meta["权威机构"] == "central_bank"
    assert meta["施行日期"] == "2024-05-01"
    assert meta["效力状态"] == "current"
    assert meta["来源"].startswith("local_pdf:")


def test_extract_pdf_metadata_abolished():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "old.pdf")
    _make_pdf(p, "本办法已被废止，不再适用")
    meta = extract_pdf_metadata(p)
    assert meta["效力状态"] == "abolished"


def test_extract_pdf_metadata_missing_file_fallback():
    # 不存在的文件 → 兜底 dict，不抛异常
    meta = extract_pdf_metadata(os.path.join(tempfile.mkdtemp(), "nope.pdf"))
    assert meta["效力状态"] == "unknown"
    assert meta["_meta_source"] == "pdf_fallback"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
