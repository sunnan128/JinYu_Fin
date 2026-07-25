# -*- coding: utf-8 -*-
"""金融 RAG 三层幻觉抑制守卫（hallucination_guard）。

本模块为【纯新增、零第三方依赖（仅标准库）】的守卫组件，
不修改任何现有业务代码，可独立被检索/生成链路调用。

三层职责
========
L1  检索端权威度过滤
    在"检索召回后、喂给 LLM 前"生效。消费每个 chunk.metadata 中的
    权威机构 / 效力状态 / 施行日期 / 排序键 / 令号：
      - 剔除 效力状态 为"已废止/失效"的候选块；
      - 对同法规（相同 令号/文件名）的多个版本按 排序键/施行日期 去重，只保留最新版；
      - 按权威机构权重排序，优先权威来源。
    从源头减少"喂给模型的就是脏数据"导致的幻觉。

L2  Prompt 强约束
    在"让 LLM 生成答案"这一步生效。仅负责拼装 system prompt 字符串
    （不调用任何模型），硬性约束：只能依据检索片段作答；片段无依据必须
    答"未找到相关信息"；严禁编造条款/编号；发现已废止条款须提示时效风险。

L3  生成后校验
    在"答案已生成、尚未返回用户"这一步生效。将答案文本与 superseded.json
    （废止条款对照表）比对，若答案引用了已废止条款，则判定 blocked=True，
    返回 action（默认 regenerate 要求依据现行有效条款重答）并打标命中项。
    这是最后一道兜底。

chunk 输入兼容形态
==================
  - DocumentChunk（backend.utils.document_parser.DocumentChunk）
  - 任意带 .metadata 属性的对象
  - dict：metadata 在 obj["metadata"]，正文在 obj["content"]/obj["text"]

superseded.json 结构
=====================
{
  "version": 1,
  "entries": [
    {
      "id": "commercial_bank_ltv_75",
      "law": "中华人民共和国商业银行法",
      "clause": "第三十九条（存贷比不得超过75%）",
      "keywords": ["存贷比", "75%"],
      "status": "abolished",
      "strict": false,
      "replaced_by": "商业银行流动性风险管理办法",
      "issued_date": "1995-07-01",
      "abolished_by": "《全国人民代表大会常务委员会关于修改〈中华人民共和国商业银行法〉的决定》",
      "abolished_date": "2015-08-29",
      "source": "..."
    }
  ]
}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

# ---------- 路径与常量 ----------
DEFAULT_SUPERSEDED_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "superseded.json"
)

# 效力状态取值映射（小写比较）
ABOLISHED_TOKENS = {
    "abolished", "已废止", "repealed", "失效", "invalid", "作废", "被废止",
}
CURRENT_TOKENS = {
    "current", "现行有效", "有效", "生效", "valid",
}

# 权威机构 -> 权威度权重（数值越大越权威；用于 L1 排序，可选增强）
AUTHORITY_WEIGHTS = {
    "全国人大": 100, "全国人民代表大会常务委员会": 100, "国务院": 95,
    "central_bank": 90, "中央银行": 90, "中国人民银行": 90,
    "证监会": 88, "中国证券监督管理委员会": 88,
    "银保监会": 88, "金融监管总局": 88, "国家金融监督管理总局": 88,
    "财政部": 85, "国家税务总局": 80,
}
DEFAULT_AUTHORITY_WEIGHT = 50


# ---------- 通用：从多种 chunk 形态提取 metadata / content ----------
def _get_metadata(chunk: Any) -> Dict[str, Any]:
    if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
        return chunk.metadata
    if isinstance(chunk, dict):
        meta = chunk.get("metadata")
        if isinstance(meta, dict):
            return meta
        return chunk
    return {}


def _get_content(chunk: Any) -> str:
    if hasattr(chunk, "content"):
        return chunk.content or ""
    if isinstance(chunk, dict):
        return chunk.get("content") or chunk.get("text") or ""
    return str(chunk)


# ---------- L1：检索端权威度过滤 ----------
def _is_abolished(meta: Dict[str, Any]) -> bool:
    status = str(meta.get("效力状态", meta.get("is_current", ""))).strip()
    if not status:
        return False
    low = status.lower()
    if low in ABOLISHED_TOKENS:
        return True
    if low in CURRENT_TOKENS:
        return False
    return any(tok in status for tok in ("废止", "失效", "作废", "abolish", "repeal", "invalid"))


def _doc_id(meta: Dict[str, Any]) -> str:
    """同法规版本去重标识：令号优先，否则文件名（去扩展名）。"""
    order = meta.get("令号") or meta.get("doc_order")
    if order:
        return str(order)
    fn = meta.get("filename") or meta.get("source") or ""
    return os.path.splitext(str(fn))[0]


def _version_key(meta: Dict[str, Any]):
    """用于"取最新版"的可比较键：(排序键数值, 施行日期字符串)。"""
    raw = meta.get("排序键") or meta.get("doc_order_key") or ""
    try:
        order_val = float(str(raw))
    except (TypeError, ValueError):
        order_val = 0.0
    date = meta.get("施行日期") or meta.get("effective_date") or ""
    return (order_val, str(date))


@dataclass
class L1Result:
    kept: List[Any] = field(default_factory=list)
    removed_abolished: List[Any] = field(default_factory=list)
    deduplicated: List[Any] = field(default_factory=list)


def filter_by_authority(
    chunks: Sequence[Any],
    keep_top: Optional[int] = None,
    enable_dedup: bool = True,
) -> L1Result:
    """L1 检索端权威度过滤。

    1. 剔除 效力状态 为已废止/失效的 chunk；
    2. （可选）对相同法规的多个版本按 排序键/施行日期 去重，仅保留最新版；
    3. 按权威机构权重稳定排序，优先权威来源。
    返回原 chunk 对象引用（不复制内容），调用方据此过滤后再喂给 LLM。
    """
    kept: List[Any] = []
    removed_abolished: List[Any] = []
    for chunk in chunks:
        meta = _get_metadata(chunk)
        if _is_abolished(meta):
            removed_abolished.append(chunk)
        else:
            kept.append(chunk)

    deduplicated: List[Any] = []
    if enable_dedup:
        # 仅当同一法规（did）存在【多个不同版本】(vk) 时才去重，只保留最新版。
        # 修复：单文件多 chunk（无令号、全部 vk 相同）属于“同一版本多个片段”，
        # 不应被压缩成 1 个 chunk，否则会破坏检索召回。
        from collections import defaultdict
        versions: Dict[str, set] = defaultdict(set)
        for chunk in kept:
            did = _doc_id(_get_metadata(chunk))
            if did:
                versions[did].add(_version_key(_get_metadata(chunk)))
        multi_version_dids = {d for d, vks in versions.items() if len(vks) > 1}

        if multi_version_dids:
            best: Dict[str, Any] = {}
            best_key: Dict[str, tuple] = {}
            for chunk in kept:
                meta = _get_metadata(chunk)
                did = _doc_id(meta)
                if did not in multi_version_dids:
                    continue
                vk = _version_key(meta)
                if did not in best or vk > best_key[did]:
                    if did in best:
                        deduplicated.append(best[did])
                    best[did] = chunk
                    best_key[did] = vk
            seen: set = set()
            deduped_kept: List[Any] = []
            for chunk in kept:
                meta = _get_metadata(chunk)
                did = _doc_id(meta)
                if did not in multi_version_dids:
                    deduped_kept.append(chunk)
                elif best.get(did) is chunk and did not in seen:
                    deduped_kept.append(chunk)
                    seen.add(did)
            kept = deduped_kept

    def _auth_weight(chunk: Any) -> int:
        meta = _get_metadata(chunk)
        auth = str(meta.get("权威机构", meta.get("authority", ""))).strip()
        return AUTHORITY_WEIGHTS.get(auth, DEFAULT_AUTHORITY_WEIGHT)

    kept.sort(key=_auth_weight, reverse=True)

    if keep_top is not None:
        kept = kept[:keep_top]

    return L1Result(kept=kept, removed_abolished=removed_abolished, deduplicated=deduplicated)


# ---------- L2：Prompt 强约束 ----------
# 防伪约束文本（L2 核心）。既可用于组装完整 prompt，也可仅作为 system 注记注入。
L2_CONSTRAINTS_TEXT = (
    "补充约束（防伪规则）：\n"
    "1. 只能依据【检索片段】中的内容回答用户问题，不得引用片段之外的任何法规条款、"
    "条文编号或数据。\n"
    "2. 若【检索片段】中不包含回答所需信息，必须明确回答"
    "\"根据提供的资料，未找到相关信息\"，严禁编造、推测或引用外部知识。\n"
    "3. 引用条款时须标明其来源法规与条文（如片段所示），且不得对片段内容做扩展解释或改写其含义。\n"
    "4. 如发现检索片段中存在已废止或已失效的条款，应在回答中提示该条款的时效性风险，"
    "但不得将其作为有效依据。\n"
)


def build_constrained_system_note() -> str:
    """返回可直接拼接到 system prompt 的 L2 强约束文本（不调用模型）。"""
    return L2_CONSTRAINTS_TEXT


def build_constrained_prompt(
    query: str,
    contexts: Sequence[Any],
    max_context_chars: int = 6000,
) -> str:
    """L2 Prompt 强约束：拼装可直接用作 system prompt 的字符串。

    不调用任何模型，仅返回约束文本 + 检索片段。由调用方拿去生成答案。
    """
    ctx_texts: List[str] = []
    for c in contexts:
        content = _get_content(c)
        if content:
            ctx_texts.append(content.strip())
    joined = "\n\n---\n\n".join(ctx_texts)
    if len(joined) > max_context_chars:
        joined = joined[:max_context_chars] + "\n…（检索片段过长已截断）"

    prompt = (
        "你是金融合规问答助手。请严格遵守以下规则：\n"
        + L2_CONSTRAINTS_TEXT
        + f"\n【用户问题】\n{query}\n\n"
        f"【检索片段】\n{joined if joined else '（无检索片段）'}\n"
    )
    return prompt


# ---------- L3：生成后校验 ----------
@dataclass
class SupersededHit:
    entry_id: str
    law: str
    clause: str
    keyword: str
    replaced_by: str
    abolished_by: str = ""     # 明文废止/修改该条款的法规或决定名称
    issued_date: str = ""      # 被废止条款所属法规的发行/施行日期
    abolished_date: str = ""   # 明文废止决定【施行日】口径
    source_url: str = ""       # 官方原文链接（供提示条跳转核实）


@dataclass
class L3Result:
    blocked: bool
    hits: List[SupersededHit] = field(default_factory=list)
    action: str = "allow"  # allow | flag | regenerate | block


def load_superseded(path: str = DEFAULT_SUPERSEDED_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _entry_keywords(entry: Dict[str, Any]) -> List[str]:
    return [str(k) for k in (entry.get("keywords") or [])]


def verify_answer(
    answer: str,
    superseded: Union[str, Dict[str, Any], None] = None,
) -> L3Result:
    """L3 生成后校验：将生成的 answer 与废止对照表比对。

    - superseded 可传路径（默认本目录 superseded.json）或已加载的 dict。
    - 若 answer 命中任意 status=abolished 的条目（出现其关键词），则 blocked=True，
      action 默认 "regenerate"（要求模型依据现行有效条款重新回答），并打标命中项。
    - 条目设 "strict": true 时，除关键词外还需同时出现 law 名称才判定命中，降低误报。
    """
    if superseded is None:
        superseded = DEFAULT_SUPERSEDED_PATH
    data = load_superseded(superseded) if isinstance(superseded, str) else superseded
    entries = data.get("entries", []) if isinstance(data, dict) else []

    answer = answer or ""
    hits: List[SupersededHit] = []
    for entry in entries:
        if entry.get("status") != "abolished":
            continue
        law = entry.get("law", "")
        clause = entry.get("clause", "")
        replaced_by = entry.get("replaced_by", "")
        strict = bool(entry.get("strict", False))
        for kw in _entry_keywords(entry):
            if kw and kw in answer:
                if strict and law and law not in answer:
                    continue
                hits.append(SupersededHit(
                    entry_id=entry.get("id", ""),
                    law=law, clause=clause, keyword=kw, replaced_by=replaced_by,
                    abolished_by=entry.get("abolished_by", ""),
                    issued_date=entry.get("issued_date", ""),
                    abolished_date=entry.get("abolished_date", ""),
                    source_url=entry.get("source_url", ""),
                ))
                break

    blocked = len(hits) > 0
    action = "regenerate" if blocked else "allow"
    return L3Result(blocked=blocked, hits=hits, action=action)
