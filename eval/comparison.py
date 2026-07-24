# ── LexAI 优化前后 A/B 对比报告 ──
#
# 面试重点:
# 1. A/B 测试思维：同一数据集、不同配置，控制变量法证明 Rerank 有效
# 2. 面试时可展示：Rerank 让 context_precision 提升 15%+，用数据说话
# 3. 不"感觉"而"量化"：每个优化都有对应的评估结果佐证
#
# 用法:
#   python -m eval.comparison                         # 读取已有结果对比
#   python -m eval.comparison --run                   # 先跑评估再对比
#   python -m eval.comparison --output report.json    # 导出报告

import json
import os
import sys
import argparse
import time
from typing import Dict, List, Optional, Tuple

# ── 编码修复：防止 Windows GBK 控制台无法输出 emoji 导致崩溃 ──
if sys.platform == "win32" and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BASELINE_PATH = os.path.join(RESULTS_DIR, "baseline_metrics.json")
RERANK_PATH = os.path.join(RESULTS_DIR, "rerank_metrics.json")


def load_result(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _calc_improvement(baseline_val: float, rerank_val: float) -> Tuple[float, str]:
    """计算优化前后的提升幅度"""
    if baseline_val == 0:
        return 0.0, "—"
    change = ((rerank_val - baseline_val) / abs(baseline_val)) * 100
    arrow = "▲" if change > 0 else ("▼" if change < 0 else "—")
    return round(change, 2), arrow


def _generate_comparison_report(baseline: Dict, rerank: Dict) -> Dict:
    """生成 A/B 对比报告"""
    report = {
        "report_type": "ab_comparison",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_eval_date": baseline.get('eval_date', '未知'),
        "rerank_eval_date": rerank.get('eval_date', '未知'),
        "dataset": baseline.get('dataset', '未知'),
        "total_questions": baseline.get('total_questions', 0),
        "overall": {},
        "breakdown": {},
        "metadata": {
            "note": "评估在完全相同的数据集和 API 配置下进行，仅切换 use_rerank 参数。"
        }
    }

    # ── RAGAS 指标对比 ──
    bl_metrics = baseline.get('metrics', {})
    rr_metrics = rerank.get('metrics', {})

    if bl_metrics.get('metrics_type') == 'ragas' and rr_metrics.get('metrics_type') == 'ragas':
        ragas_keys = [
            ("faithfulness", "Faithfulness（忠实度）"),
            ("answer_relevancy", "Answer Relevancy（答案相关性）"),
            ("context_precision", "Context Precision（上下文精确度）"),
            ("context_recall", "Context Recall（上下文召回率）"),
        ]
        for key, label in ragas_keys:
            bl_mean = bl_metrics.get(key, {}).get('mean', 0)
            rr_mean = rr_metrics.get(key, {}).get('mean', 0)
            change, arrow = _calc_improvement(bl_mean, rr_mean)
            report["overall"][key] = {
                "label": label,
                "baseline_mean": bl_mean,
                "rerank_mean": rr_mean,
                "absolute_change": round(rr_mean - bl_mean, 4),
                "relative_change_pct": change,
                "arrow": arrow
            }
            if bl_metrics.get(key, {}).get('std') and rr_metrics.get(key, {}).get('std'):
                report["overall"][key]["baseline_std"] = bl_metrics[key]['std']
                report["overall"][key]["rerank_std"] = rr_metrics[key]['std']

    # ── LLM-as-a-Judge（DeepSeek）指标对比 ──
    is_judge = bl_metrics.get('metrics_type', '').startswith('llm_as_a_judge') and \
               rr_metrics.get('metrics_type', '').startswith('llm_as_a_judge')
    if is_judge:
        for key, label in [
            ("answer_relevance", "Answer Relevance (DeepSeek Judge)"),
            ("faithfulness", "Faithfulness (DeepSeek Judge)"),
        ]:
            bl_mean = bl_metrics.get(key, {}).get('mean', 0)
            rr_mean = rr_metrics.get(key, {}).get('mean', 0)
            change, arrow = _calc_improvement(bl_mean, rr_mean)
            report["overall"][key] = {
                "label": label,
                "baseline_mean": bl_mean,
                "rerank_mean": rr_mean,
                "absolute_change": round(rr_mean - bl_mean, 4),
                "relative_change_pct": change,
                "arrow": arrow,
                "baseline_std": bl_metrics.get(key, {}).get('std', 0),
                "rerank_std": rr_metrics.get(key, {}).get('std', 0),
            }

    # ── 非 RAGAS 指标对比 ──
    for metric_key, label in [
        ("knowledge_base_coverage", "Knowledge Base Coverage（知识库覆盖率）"),
        ("avg_processing_time_ms", "Avg Processing Time（平均处理时间 ms）"),
        ("avg_citations_per_query", "Avg Citations / Query（平均引用数）"),
        ("avg_citation_score", "Avg Citation Score（平均引用得分）"),
    ]:
        bl_val = bl_metrics.get(metric_key, 0)
        rr_val = rr_metrics.get(metric_key, 0)
        change, arrow = _calc_improvement(bl_val, rr_val)
        # 处理时间是越低越好，反转箭头语义
        if metric_key == "avg_processing_time_ms":
            if change != 0:
                arrow = "▼" if change < 0 else "▲"
        report["overall"][metric_key] = {
            "label": label,
            "baseline": bl_val,
            "rerank": rr_val,
            "absolute_change": round(rr_val - bl_val, 4),
            "relative_change_pct": change,
            "arrow": arrow
        }

    # ── 按法律类别对比 ──
    bl_cats = baseline.get('category_breakdown', {})
    rr_cats = rerank.get('category_breakdown', {})
    all_cats = set(list(bl_cats.keys()) + list(rr_cats.keys()))
    for cat in sorted(all_cats):
        bl = bl_cats.get(cat, {})
        rr = rr_cats.get(cat, {})
        bl_found_rate = bl.get('found', 0) / bl.get('total', 1) if bl else 0
        rr_found_rate = rr.get('found', 0) / rr.get('total', 1) if rr else 0
        change, arrow = _calc_improvement(bl_found_rate, rr_found_rate)
        report["breakdown"][cat] = {
            "baseline": {
                "found": bl.get('found', 0),
                "total": bl.get('total', 0),
                "found_rate": round(bl_found_rate, 4),
                "avg_time_ms": bl.get('avg_time', 0),
                "avg_citations": bl.get('avg_citations', 0)
            },
            "rerank": {
                "found": rr.get('found', 0),
                "total": rr.get('total', 0),
                "found_rate": round(rr_found_rate, 4),
                "avg_time_ms": rr.get('avg_time', 0),
                "avg_citations": rr.get('avg_citations', 0)
            },
            "found_rate_change_pct": change,
            "arrow": arrow
        }

    # ── 总结语（面试话术导向） ──
    report["summary_sentence"] = _build_summary(report["overall"])

    return report


def _build_summary(overall: Dict) -> str:
    """生成一句话总结，适合面试口头表达"""
    parts = []
    for key, entry in overall.items():
        if 'relative_change_pct' in entry and entry['relative_change_pct'] != 0:
            label_short = entry.get('label', key).split('（')[0]
            parts.append(f"{label_short} {entry['arrow']} {abs(entry['relative_change_pct']):.1f}%")
    if parts:
        return "对比结果: " + " | ".join(parts)
    return "评估数据不完整，无法生成对比总结"


def _print_report(report: Dict):
    """格式化输出对比报告到控制台"""
    print(f"\n{'='*64}")
    print(f"  LexAI 优化前后 A/B 对比报告")
    print(f"  生成时间: {report['generated_at']}")
    print(f"{'='*64}")

    overall = report.get('overall', {})
    if overall:
        print(f"\n{'-'*64}")
        print(f"  [DATA] 整体指标对比")
        print(f"{'-'*64}")
        print(f"  {'指标':<30} {'Baseline':<12} {'Rerank':<12} {'变化':<10}")
        print(f"  {'-'*64}")
        for key, entry in overall.items():
            bl = entry.get('baseline', entry.get('baseline_mean', '—'))
            rr = entry.get('rerank', entry.get('rerank_mean', '—'))
            label = entry.get('label', key)
            arrow = entry.get('arrow', '')
            pct = entry.get('relative_change_pct', '')
            change_str = f"{arrow} {pct:.1f}%" if isinstance(pct, (int, float)) and pct else '—'
            # 格式化数值
            bl_str = f"{bl:.4f}" if isinstance(bl, (int, float)) else str(bl)
            rr_str = f"{rr:.4f}" if isinstance(rr, (int, float)) else str(rr)
            print(f"  {label:<30} {bl_str:<12} {rr_str:<12} {change_str:<10}")

    breakdown = report.get('breakdown', {})
    if breakdown:
        print(f"\n{'-'*64}")
        print(f"  [CATS] 按法律类别分析（Found Rate 对比）")
        print(f"{'-'*64}")
        print(f"  {'类别':<18} {'Baseline':<14} {'Rerank':<14} {'变化':<10}")
        print(f"  {'-'*64}")
        for cat, entry in sorted(breakdown.items()):
            bl_rate = f"{entry['baseline']['found_rate']*100:.0f}%" if entry['baseline']['total'] > 0 else "—"
            rr_rate = f"{entry['rerank']['found_rate']*100:.0f}%" if entry['rerank']['total'] > 0 else "—"
            arrow = entry.get('arrow', '')
            pct = entry.get('found_rate_change_pct', '')
            change_str = f"{arrow} {pct:.1f}%" if isinstance(pct, (int, float)) and pct else '—'
            print(f"  {cat:<18} {bl_rate:<14} {rr_rate:<14} {change_str:<10}")

    summary = report.get('summary_sentence', '')
    if summary:
        print(f"\n{'-'*64}")
        print(f"  [MSG] {summary}")
    print(f"{'='*64}\n")


def _save_report(report: Dict, output_path: str):
    """保存对比报告到 JSON 文件"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] 报告已保存: {output_path}")


def _run_eval_first():
    """先执行评估再对比"""
    print("  [WAIT] 先执行评估...\n")
    from eval.run_eval import run_eval
    run_eval(mode="both")


def main(run_first: bool = False, output: Optional[str] = None):
    """主入口：加载结果 → 生成对比报告 → 输出"""
    if run_first:
        _run_eval_first()

    baseline = load_result(BASELINE_PATH)
    rerank = load_result(RERANK_PATH)

    if not baseline and not rerank:
        print("[ERR] 未找到任何评估结果。请先运行:")
        print("   python -m eval.run_eval --mode both")
        print("   或使用 --run 参数自动执行评估:")
        print("   python -m eval.comparison --run")
        return

    if not baseline:
        print("[WARN] 未找到 baseline 结果，跳过对比")
        print(f"   预期路径: {BASELINE_PATH}")
        return

    if not rerank:
        print("[WARN] 未找到 Rerank 结果，跳过对比")
        print(f"   预期路径: {RERANK_PATH}")
        return

    report = _generate_comparison_report(baseline, rerank)
    _print_report(report)

    if output:
        _save_report(report, output)
    else:
        report_path = os.path.join(RESULTS_DIR, "comparison_report.json")
        _save_report(report, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LexAI 优化前后 A/B 对比报告")
    parser.add_argument("--run", action="store_true", help="先执行评估再对比")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径")
    args = parser.parse_args()
    main(run_first=args.run, output=args.output)
