# ── L3 幻觉抑制前端提示条 ──
# 职责：根据后端 /query 返回的 guard 标记（GuardInfo）生成 ⚠️ 时效风险 提示 HTML。
# 设计为纯函数、不依赖 streamlit，便于单元测试，也避免在主前端文件中堆砌 HTML 拼接逻辑。

def build_guard_banner_html(guard) -> str:
    """根据 guard 字段生成时效风险提示条的 HTML 字符串。

    参数 guard 预期结构（来自 QueryResponse.guard / GuardInfo）：
        {
          "blocked": bool,
          "action": "allow" | "regenerate",
          "hits": [
            {
              "law": str, "clause": str, "keyword": str, "replaced_by": str,
              "abolished_by": str, "issued_date": str, "abolished_date": str,
              "source_url": str
            },
            ...
          ]
        }

    行为：
        - guard 缺失 或 blocked 为假 -> 返回空串（不打扰正常问答）。
        - blocked 为真且有命中明细 -> 列出已废止/失效条款，提示以现行有效规定为准。
        - blocked 为真但无明细（防御性）-> 给一条通用时效风险提醒。

    返回：可直接 st.markdown(..., unsafe_allow_html=True) 的 HTML 片段。
    """
    if not guard:
        return ""
    if not guard.get("blocked", False):
        return ""

    hits = guard.get("hits") or []
    if not hits:
        # 防御性：理论上 blocked 为真时必有 hits，但此处兜底给通用提示
        return (
            '<div class="guard-banner">'
            '<div class="guard-title">⚠️ 时效风险提醒</div>'
            '<div class="guard-body">系统检测到本次回答可能涉及已废止或失效的法规条款，'
            '已自动执行时效校验。请务必以现行有效规定为准，谨慎核实后再做决策。</div>'
            '</div>'
        )

    hit_items = []
    for h in hits:
        law = (h.get("law") or "未知法规").strip()
        clause = (h.get("clause") or "").strip()
        keyword = (h.get("keyword") or "").strip()
        replaced = (h.get("replaced_by") or "现行有效规定").strip()
        abolished_by = (h.get("abolished_by") or "").strip()
        issued_date = (h.get("issued_date") or "").strip()
        abolished_date = (h.get("abolished_date") or "").strip()
        source_url = (h.get("source_url") or "").strip()

        detail_lines = []
        if issued_date:
            detail_lines.append(f'<span class="guard-detail">📅 发行日期：{issued_date}</span>')
        if abolished_by:
            detail_lines.append(f'<span class="guard-detail">📜 明文废止依据：{abolished_by}</span>')
        if abolished_date:
            detail_lines.append(f'<span class="guard-detail">📅 废止日期（决定施行日）：{abolished_date}</span>')
        if source_url:
            safe_url = source_url.replace('"', "%22")
            detail_lines.append(
                f'<span class="guard-detail">🔗 <a href="{safe_url}" target="_blank" '
                f'rel="noopener noreferrer">查看官方原文</a></span>'
            )
        detail_html = "".join(detail_lines)

        hit_items.append(
            f"<li><strong>{law}{clause}</strong>：命中关键词「{keyword}」，"
            f"该条款可能已废止/失效，请以 <strong>{replaced}</strong> 为准"
            + (f'<div class="guard-details">{detail_html}</div>' if detail_html else "")
            + "</li>"
        )
    hit_html = "".join(hit_items)

    return (
        '<div class="guard-banner">'
        '<div class="guard-title">⚠️ 时效风险提醒</div>'
        '<div class="guard-body">系统检测到本次回答引用/涉及了'
        '<span class="guard-em">已废止或失效的法规条款</span>，'
        '已自动执行时效校验并尽量以现行有效规定重新生成答案。'
        '以下为命中的失效条款及其时效依据，请务必核实：</div>'
        f'<ul class="guard-list">{hit_html}</ul>'
        '</div>'
    )
