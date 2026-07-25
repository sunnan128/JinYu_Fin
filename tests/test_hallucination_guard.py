# -*- coding: utf-8 -*-
"""hallucination_guard 三层幻觉抑制 + document_parser 回归测试。"""
import os
import tempfile

import document_parser as dp
import hallucination_guard as hg


def _chunk(content, **meta):
    return dp.DocumentChunk(content=content, metadata=meta)


# ============ L1 检索端权威度过滤 ============
def test_l1_removes_abolished():
    chunks = [
        _chunk("现行有效的存款规定", 效力状态="current", 权威机构="中国人民银行",
               排序键="1", 令号="令A"),
        _chunk("已废止的旧规定", 效力状态="已废止", 权威机构="中国人民银行",
               排序键="1", 令号="令B"),
    ]
    res = hg.filter_by_authority(chunks)
    assert len(res.kept) == 1
    assert res.kept[0].content == "现行有效的存款规定"
    assert len(res.removed_abolished) == 1
    assert res.removed_abolished[0].content == "已废止的旧规定"


def test_l1_dedup_keeps_latest_version():
    chunks = [
        _chunk("旧版反洗钱法条款", 效力状态="current", 令号="反洗钱法",
               排序键="2006", 施行日期="2006-10-31"),
        _chunk("新版反洗钱法条款", 效力状态="current", 令号="反洗钱法",
               排序键="2024", 施行日期="2025-01-01"),
    ]
    res = hg.filter_by_authority(chunks)
    assert len(res.kept) == 1
    assert res.kept[0].content == "新版反洗钱法条款"
    assert len(res.deduplicated) == 1


def test_l1_keeps_current_when_no_dup():
    chunks = [
        _chunk("规定A", 效力状态="current", 令号="法1", 排序键="1"),
        _chunk("规定B", 效力状态="current", 令号="法2", 排序键="2"),
    ]
    res = hg.filter_by_authority(chunks)
    assert len(res.kept) == 2
    assert len(res.deduplicated) == 0


def test_l1_dict_chunk_compat():
    chunks = [
        {"content": "有效条款", "metadata": {"效力状态": "current", "令号": "d1"}},
        {"content": "废止条款", "metadata": {"效力状态": "abolished", "令号": "d2"}},
    ]
    res = hg.filter_by_authority(chunks)
    assert len(res.kept) == 1
    assert res.kept[0]["content"] == "有效条款"


def test_l1_single_file_multi_chunk_not_collapsed():
    """修复回归：同一文件（无令号、无版本差异）的多个 chunk 不应被去重压缩成 1 块。"""
    chunks = [
        _chunk("第一条 内容A", 效力状态="current", filename="某办法.md"),
        _chunk("第二条 内容B", 效力状态="current", filename="某办法.md"),
        _chunk("第三条 内容C", 效力状态="current", filename="某办法.md"),
    ]
    res = hg.filter_by_authority(chunks)
    assert len(res.kept) == 3
    assert len(res.deduplicated) == 0


# ============ L2 Prompt 强约束 ============
def test_l2_prompt_has_constraints_and_context():
    prompt = hg.build_constrained_prompt(
        "存贷比限制是多少？", ["根据XX法，存贷比不得超过75%"]
    )
    assert "只能依据" in prompt
    assert "检索片段" in prompt
    assert "未找到相关信息" in prompt
    assert "存贷比不得超过75%" in prompt
    assert "存贷比限制是多少？" in prompt


# ============ L3 生成后校验 ============
def test_l3_blocks_abolished_reference():
    answer = "根据《商业银行法》，存贷比不得超过75%。"
    res = hg.verify_answer(answer)
    assert res.blocked is True
    assert res.action == "regenerate"
    assert any(h.entry_id == "commercial_bank_ltv_75" for h in res.hits)


def test_l3_allows_clean_answer():
    answer = "根据现行《商业银行流动性风险管理办法》，银行需满足流动性覆盖率要求。"
    res = hg.verify_answer(answer)
    assert res.blocked is False
    assert res.action == "allow"
    assert res.hits == []


def test_l3_strict_mode_requires_law_name():
    tmp = {
        "entries": [
            {"id": "x", "law": "某法", "clause": "第1条",
             "keywords": ["某关键词"], "status": "abolished", "strict": True}
        ]
    }
    # 仅含关键词、不含法名 -> strict 不命中
    assert hg.verify_answer("这里提到某关键词但没说法名", tmp).blocked is False
    # 含法名 -> 命中
    r = hg.verify_answer("某法第1条规定某关键词", tmp)
    assert r.blocked is True
    assert r.hits[0].entry_id == "x"


def test_l3_skips_non_abolished_entries():
    tmp = {
        "entries": [
            {"id": "y", "law": "某有效法", "keywords": ["有效词"],
             "status": "current", "strict": False}
        ]
    }
    assert hg.verify_answer("文本包含有效词", tmp).blocked is False


# ============ 回归：document_parser 现有主要功能无异常 ============
def test_extract_front_matter_chinese_keys_and_no_misdelete():
    text = (
        "# 权威机构: 中国人民银行\n"
        "# 令号: 国务院令第768号\n"
        "# 排序键: 768\n"
        "# 施行日期: 2025-01-01\n"
        "# 效力状态: current\n\n"
        "# 第一章 总则\n"
        "第一条 为了规范XX，制定本法。\n"
    )
    cleaned, meta = dp.extract_front_matter(text)
    assert meta.get("权威机构") == "中国人民银行"
    assert meta.get("效力状态") == "current"
    assert meta.get("排序键") == "768"
    # 真章节标题不应被误删
    assert "# 第一章 总则" in cleaned
    assert "第一条 为了规范XX，制定本法。" in cleaned


def test_extract_front_matter_no_front_matter_passthrough():
    text = "# 第一章 总则\n第一条 内容\n"
    cleaned, meta = dp.extract_front_matter(text)
    assert meta == {}
    assert cleaned == text


def test_parse_markdown_boundary_aware_no_inline_mis_split():
    md = (
        "# 效力状态: current\n\n"
        "本法第五十三条 不得从事下列活动。\n\n"
        "第五十四条 违反前款规定的处罚。\n"
    )
    p = os.path.join(tempfile.gettempdir(), "tmp_hg_md_test.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(md)
    chunks = dp.DocumentParser.parse_markdown(p)
    bodies = [c.content for c in chunks]
    # 应切成 2 块：一块含"本法第五十三条"（行内引用，不单独成块），一块含"第五十四条"
    assert len(chunks) == 2
    assert any("本法第五十三条" in b for b in bodies)
    # "本法第五十三条" 不应作为独立条款起点被误切
    assert not any(b.strip().startswith("第五十三条") for b in bodies)
    os.remove(p)
