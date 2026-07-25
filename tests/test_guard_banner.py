# ── 前端 ⚠️ 时效风险 提示条（L3 幻觉抑制打标展示）单测 ──
# 只测纯函数 build_guard_banner_html（无 streamlit 依赖），确保：
#   - guard 缺失/未拦截 -> 不打扰（返回空串）
#   - 命中已废止条款 -> 列出 law/clause/keyword/replaced_by
#   - 兜底：blocked 但无明细 -> 通用提示
import os
import sys

_FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)

from guard_banner import build_guard_banner_html


def test_none_guard_returns_empty():
    assert build_guard_banner_html(None) == ""
    assert build_guard_banner_html({}) == ""


def test_blocked_false_returns_empty():
    assert build_guard_banner_html({"blocked": False, "action": "allow", "hits": []}) == ""


def test_blocked_with_hits_lists_clauses():
    guard = {
        "blocked": True,
        "action": "regenerate",
        "hits": [
            {
                "law": "《中华人民共和国商业银行法》",
                "clause": "第三十九条（存贷比75%）",
                "keyword": "存贷比",
                "replaced_by": "《商业银行流动性风险管理办法》",
            }
        ],
    }
    html = build_guard_banner_html(guard)
    assert "guard-banner" in html
    assert "时效风险" in html
    assert "存贷比" in html
    assert "《商业银行流动性风险管理办法》" in html
    assert "<li>" in html


def test_blocked_no_hits_generic():
    html = build_guard_banner_html({"blocked": True, "action": "regenerate", "hits": []})
    assert "guard-banner" in html
    assert "时效风险" in html


def test_blocked_shows_abolished_details():
    guard = {
        "blocked": True,
        "action": "regenerate",
        "hits": [
            {
                "law": "《中华人民共和国商业银行法》",
                "clause": "第三十九条（存贷比75%）",
                "keyword": "存贷比",
                "replaced_by": "《商业银行流动性风险管理办法》",
                "abolished_by": "《全国人民代表大会常务委员会关于修改〈中华人民共和国商业银行法〉的决定》（2015年8月29日通过）",
                "issued_date": "1995-07-01",
                "abolished_date": "2015-10-01",
                "source_url": "https://flk.npc.gov.cn/",
            }
        ],
    }
    html = build_guard_banner_html(guard)
    assert "明文废止依据" in html
    assert "发行日期" in html
    assert "废止日期" in html
    assert "1995-07-01" in html
    assert "2015-10-01" in html
    assert "商业银行流动性风险管理办法" in html
    # 官方原文链接
    assert "查看官方原文" in html
    assert 'href="https://flk.npc.gov.cn/"' in html
    assert 'target="_blank"' in html
    # 缺失字段不应产生空标签
    assert "发行日期：" not in html.replace("发行日期：1995-07-01", "")


def test_blocked_missing_details_no_empty_labels():
    guard = {
        "blocked": True,
        "action": "regenerate",
        "hits": [
            {
                "law": "《某法规》",
                "clause": "第X条",
                "keyword": "关键词",
                "replaced_by": "现行规定",
                # abolished_by / issued_date / abolished_date / source_url 均缺失
            }
        ],
    }
    html = build_guard_banner_html(guard)
    assert "明文废止依据" not in html
    assert "发行日期" not in html
    assert "废止日期" not in html
    assert "查看官方原文" not in html
    assert "某法规" in html
