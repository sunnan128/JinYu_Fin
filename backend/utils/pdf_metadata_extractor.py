# -*- coding: utf-8 -*-
"""PDF 元数据抽取（B 方案）：让上传/灌库的 PDF 法规也能携带时效元数据。

背景
====
金语 AI 的三层幻觉抑制（L1 检索端权威度过滤）依赖每个 chunk 的元数据：
权威机构 / 令号 / 施行日期 / 效力状态 / 排序键。
爬虫产线（finance_rag_data/crawl_regulations.py）会把元数据写成 .md 顶部的
front-matter，但用户**直接上传的 PDF** 没有这层头，导致 document_parser.parse_pdf
只给 chunk 挂了 filename，L1 对 PDF 完全空转（抽不到废止/版本信息）。

本模块从 PDF 正文中用「与爬虫同源」的正则 heuristic 抽取这些字段，并与 .md
front-matter 使用【完全相同的键名】（权威机构 / 令号 / 施行日期 / 效力状态 /
排序键），因此抽取结果可直接并入 chunk.metadata，被 hallucination_guard 原样
消费——守卫代码零改动。

设计要点
========
- 仅依赖 PyMuPDF（fitz，已在 .venv），无新第三方依赖。
- 抽取都是 heuristic，故额外返回 _meta_source / _meta_confidence 透明标注，
  绝不伪造"高置信度"（面试可诚实说明：PDF 元数据是正文正则估的，关键字段
  建议人工核对 CSV）。
- 抽不到的字段留空串，尤其 效力状态 默认 unknown —— L1 的 _is_abolished 对
  unknown 返回 False，即"宁可漏拦、绝不误拦"，安全不误伤有效内容。
"""

import os
import re

try:
    import fitz
except ImportError:
    fitz = None


# ---------- 发布机关：中文名 -> 与爬虫 AUTHORITY_MAP 一致的规范键 ----------
# （爬虫按域名判，PDF 无域名，故按正文出现的中文机关名判，落到同一套规范键，
#  保证与 .md 语料库 schema 完全一致，L1 权威度排序可直接生效。）
AGENCY_NAME_TO_KEY = [
    ("中国人民银行", "central_bank"),
    ("中央银行", "central_bank"),
    ("国家金融监督管理总局", "nfra"),
    ("金融监管总局", "nfra"),
    ("中国银行保险监督管理委员会", "nfra"),
    ("中国银行业监督管理委员会", "nfra"),
    ("中国保险监督管理委员会", "nfra"),
    ("中国证券监督管理委员会", "csrc"),
    ("证监会", "csrc"),
    ("上海证券交易所", "exchange"),
    ("深圳证券交易所", "exchange"),
    ("证券交易所", "exchange"),
    ("国务院", "gov_other"),
    ("全国人民代表大会常务委员会", "gov_other"),
    ("全国人民代表大会", "gov_other"),
]

# 令号正则（与爬虫 extract_doc_order 同源思路：国务院令 / 央行令 / 证监会令…）
# 注意：发布机关后常见「令」「〔〕」「第X号」多种组合，故机关后允许可选的
# "令" 与可选的括号，再接年份/序号，兼容 令〔2024〕第1号 / 令2024第1号 / 〔2024〕第1号。
ORDER_PATTERNS = [
    r"国务院令\s*第\s*(\d+)\s*号",
    r"(?:中国人民银行|证监会|银保监会?|银行业监督管理|金融监管总局|国家金融监督管理总局)\s*令?\s*[〔【\[]?\s*(\d{4})\s*[\]〕】]?\s*第\s*(\d+)\s*号",
    r"(?:中国人民银行|证监会|银保监会?|金融监管总局)\s*[令字]\s*第\s*(\d+)\s*号",
]

# 施行日期正则（与爬虫 extract_effective_date 同源；允许"起施行"或纯"施行"）
EFFECTIVE_DATE_PATTERNS = [
    r"自\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日起施行",
    r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*起?\s*施行",
    r"（(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*(?:修订|修正)）",
    r"(?:修订|修正)于\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
]

# 效力状态（与爬虫 extract_current_status 同源逻辑：只认"自我废止"，
# 避免把"同时废止某某"误判为本法规失效）
ABOLISHED_SELF_PATTERNS = [
    r"(?:本条例|本办法|本规定|本令|本决定)\s*已?被?废止",
    r"本(?:条例|办法|规定|令|决定).{0,8}(?:予以废止|已失效|停止施行)",
]
CURRENT_PATTERNS = [
    r"现予公布",
    r"自\s*\d{4}\s*年.*起施行",
]


# ---------- 单字段抽取（纯函数，便于单测） ----------
def _detect_authority(text: str) -> str:
    for name, key in AGENCY_NAME_TO_KEY:
        if name in text:
            return key
    return "other"


def _extract_order(text: str):
    """返回 (展示串, 排序键int|None)，与爬虫格式一致。"""
    for p in ORDER_PATTERNS:
        m = re.search(p, text)
        if m:
            g = m.groups()
            if len(g) == 1:
                return f"令号第{g[0]}号", int(g[0])
            year, num = int(g[0]), int(g[1])
            return f"{year}年第{num}号", year * 1000 + num
    return "", None


def _extract_effective_date(text: str) -> str:
    for p in EFFECTIVE_DATE_PATTERNS:
        m = re.search(p, text)
        if m:
            y, mo, d = m.groups()
            return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    return ""


def _extract_status(text: str) -> str:
    for p in ABOLISHED_SELF_PATTERNS:
        if re.search(p, text):
            return "abolished"
    for p in CURRENT_PATTERNS:
        if re.search(p, text):
            return "current"
    return "unknown"


def extract_metadata_from_text(text: str, title: str = "") -> dict:
    """从法规正文（可含标题）抽取元数据，返回与 .md front-matter 同构的 dict。

    纯函数、无 IO，便于单测。字段：
        标题 / 来源 / 权威机构 / 令号 / 施行日期 / 效力状态 / 排序键
    并附加 _meta_source / _meta_confidence 透明标注（heuristic，非官方）。
    """
    combined = (title + "\n" + (text or "")) if title else (text or "")
    authority = _detect_authority(combined)
    order, order_key = _extract_order(combined)
    effective_date = _extract_effective_date(combined)
    status = _extract_status(combined)

    # 置信度：抽到令号或（明确机关+施行日期）→ medium；三者齐全 → high；否则 low
    if order and effective_date and authority != "other":
        confidence = "high"
    elif order or (authority != "other" and effective_date):
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "标题": title or "",
        "来源": "",            # PDF 无来源 URL；由 IO 封装/调用方按需填充
        "权威机构": authority,
        "令号": order,
        "施行日期": effective_date,
        "效力状态": status,
        "排序键": order_key if order_key is not None else "",
        # —— 透明标注（hallucination_guard 只读取已知键，会忽略这些）——
        "_meta_source": "pdf_heuristic",
        "_meta_confidence": confidence,
    }


def extract_pdf_metadata(pdf_path: str, max_pages: int = 5) -> dict:
    """读取 PDF，抽取前若干页正文送正则抽取；并尝试用 PDF 内建标题做标题。

    返回 extract_metadata_from_text 的同构 dict。任何异常都走兜底 dict
    （仅挂文件名 + 空字段），保证调用方（parse_pdf）绝不因元数据而崩。
    """
    fallback = {
        "标题": "",
        "来源": f"local_pdf:{os.path.basename(pdf_path)}",
        "权威机构": "other",
        "令号": "",
        "施行日期": "",
        "效力状态": "unknown",
        "排序键": "",
        "_meta_source": "pdf_fallback",
        "_meta_confidence": "low",
    }
    if fitz is None:
        return fallback

    try:
        doc = fitz.open(pdf_path)
        try:
            title = (doc.metadata or {}).get("title", "") if doc.metadata else ""
            # 元数据通常集中在首页/标题页，仅取前 max_pages 页即可，省时
            pages_text = []
            for i in range(min(max_pages, len(doc))):
                pages_text.append(doc[i].get_text())
            text = "\n".join(pages_text)
        finally:
            doc.close()

        meta = extract_metadata_from_text(text, title=title)
        meta["来源"] = f"local_pdf:{os.path.basename(pdf_path)}"
        return meta
    except Exception:
        return fallback
