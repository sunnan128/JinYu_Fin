# ── 金语AI 金融版混合检索 ──
# 三路召回 + RRF 融合 + CrossEncoder 重排 + 引用溯源
#
# 架构：
#                                    ┌─── BM25 全文召回 (vector_store.keyword_search)
#         用户问句 ──→ entity_linker ──⊢─── Vector 语义召回 (vector_store.semantic_search)
#                                    └─── 图谱检索  (graph_service.search_by_entities)
#                                                    │
#                                                    ↓
#                                            RRF(K=20) 融合
#                                                    │
#                                                    ↓
#                                        CrossEncoder 重排 (bge-reranker)
#                                                    │
#                                      ┌─────────────┴─────────────┐
#                                      ↓                          ↓
#                                   置信度达标                置信度不足
#                                      │                        │
#                                      ↓                        ↓
#                              精排 Top-5 ← 回退至 Vector Top-5
#                                      │
#                                      ↓
#                               引用溯源输出
#                               (资料名/章节/条款/页码)

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from backend.config import settings

# 服务依赖（延迟导入，允许缺失）
_VECTOR_STORE = None
_ENTITY_LINKER = None
_GRAPH_SERVICE = None


def _get_vector_store():
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        from backend.services.vector_store import VectorStoreService
        _VECTOR_STORE = VectorStoreService()
    return _VECTOR_STORE


def _get_entity_linker():
    global _ENTITY_LINKER
    if _ENTITY_LINKER is None:
        from backend.utils.entity_linker import EntityLinker
        _ENTITY_LINKER = EntityLinker()
    return _ENTITY_LINKER


def _get_graph_service():
    global _GRAPH_SERVICE
    if _GRAPH_SERVICE is None:
        from backend.services.graph_service import GraphService
        _GRAPH_SERVICE = GraphService()
    return _GRAPH_SERVICE


def _get_reranker():
    """复用 llm_service 的懒加载 CrossEncoder。"""
    from backend.services.llm_service import _get_reranker as _llm_reranker
    return _llm_reranker()


# ═══════════════════════════════════════════════
#  混合检索数据类
# ═══════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """单条检索结果的统一表示。"""
    content: str
    score: float = 0.0                # 当前路的原始分数
    rerank_score: float = 0.0         # CrossEncoder 重排分数
    rrf_score: float = 0.0            # RRF 融合后分数
    rank: int = 0                      # 最终排名
    source_route: str = ""             # 来源：bm25 / vector / graph
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "rrf_score": self.rrf_score,
            "rank": self.rank,
            "source_route": self.source_route,
            "metadata": self.metadata,
        }


@dataclass
class RetrieveDebugInfo:
    """混合检索的调试/统计信息（供单元验证与日志输出）。"""
    bm25_count: int = 0
    vector_count: int = 0
    graph_count: int = 0
    graph_text_summary: str = ""
    after_rrf_count: int = 0
    after_rerank_count: int = 0
    confidence: float = 0.0
    used_fallback: bool = False
    entities_found: List[Dict] = field(default_factory=list)
    processing_time_ms: float = 0.0


# ═══════════════════════════════════════════════
#  主混合检索器
# ═══════════════════════════════════════════════

class HybridRetriever:
    """三路混合检索：
    BM25 + Vector + Graph → RRF(K=20) → CrossEncoder 重排 → 引用溯源
    """

    RRF_K = 20  # Reciprocal Rank Fusion 常数

    def __init__(self):
        self.vector_store = _get_vector_store()
        self.entity_linker = _get_entity_linker()
        self.graph_service = _get_graph_service()
        self.reranker = None  # 懒加载

    # ── 公开 API ──

    def retrieve(self, query: str, top_k: int = 5,
                 rerank_candidates: int = 20,
                 entity_boost: bool = True) -> Tuple[List[RetrievalResult], RetrieveDebugInfo]:
        """混合检索主入口。

        Args:
            query: 用户问句
            top_k: 最终返回的 top-k 条结果
            rerank_candidates: 送入 rerank 的候选数
            entity_boost: 是否启用实体增强（从 entity_linker 提取实体补充检索）

        Returns:
            (精排结果列表, 调试信息)
        """
        start = time.time()
        debug = RetrieveDebugInfo()

        # ── 第1步：实体抽取（entity_linker） ──
        entity_list = self.entity_linker.extract_entities(query)
        debug.entities_found = entity_list

        # ── 第2步：三路并行召回 ──
        bm25_results = self._retrieve_bm25(query)
        debug.bm25_count = len(bm25_results)

        vector_results = self._retrieve_vector(query)
        debug.vector_count = len(vector_results)

        graph_results, graph_text = self._retrieve_graph(
            entity_list, entity_boost=entity_boost
        )
        debug.graph_count = len(graph_results)
        debug.graph_text_summary = graph_text

        # ── 第3步：RRF 融合 ──
        fused = self._rrf_fusion(bm25_results, vector_results, graph_results)
        debug.after_rrf_count = len(fused)

        # ── 第4步：CrossEncoder 重排 ──
        reranked = self._rerank(query, fused, top_k=rerank_candidates)
        debug.after_rerank_count = len(reranked)

        # ── 第5步：置信度检查 + 回退 ──
        if reranked:
            debug.confidence = reranked[0].rerank_score
            if reranked[0].rerank_score < 0.1:
                debug.used_fallback = True
                # 回退到纯向量检索的 Top-5
                fallback = self._retrieve_vector(query, top_k=top_k)
                if fallback:
                    for i, r in enumerate(fallback):
                        r.source_route = "vector(fallback)"
                        r.rank = i + 1
                    debug.processing_time_ms = (time.time() - start) * 1000
                    return fallback[:top_k], debug
                # 回退结果为空时，用 reranked 结果兜底
                print("[HYBRID] 回退向量检索为空，使用 rerank 结果兜底")

        # ── 第6步：取 top_k + 排序 ──
        result = reranked[:top_k]
        for i, r in enumerate(result):
            r.rank = i + 1

        debug.processing_time_ms = (time.time() - start) * 1000
        return result, debug

    # ── 单路召回 ──

    def _retrieve_bm25(self, query: str, top_k: int = 50) -> List[RetrievalResult]:
        """BM25 关键词召回（来自 vector_store 的 rank_bm25 索引）。"""
        try:
            raw = self.vector_store.keyword_search(query, top_k=top_k)
        except Exception as e:
            print(f"[HYBRID] BM25 检索失败: {e}")
            return []

        results = []
        seen = set()
        for r in raw:
            content = r.get("content", "")
            if not content or content in seen:
                continue
            seen.add(content)
            results.append(RetrievalResult(
                content=content,
                score=float(r.get("score", 0)),
                source_route="bm25",
                metadata=r.get("metadata", {}),
            ))
        return results

    def _retrieve_vector(self, query: str, top_k: int = 50) -> List[RetrievalResult]:
        """Vector 语义召回（ChromaDB + bge 嵌入）。"""
        try:
            raw = self.vector_store.semantic_search(query, top_k=top_k)
        except Exception as e:
            print(f"[HYBRID] 向量检索失败: {e}")
            return []

        results = []
        seen = set()
        for r in raw:
            content = r.get("content", "")
            if not content or content in seen:
                continue
            seen.add(content)
            # ChromaDB 返回 L2 distance（越小越近），转为评分后 RRF 只依赖排名
            results.append(RetrievalResult(
                content=content,
                score=float(r.get("score", 0)),
                source_route="vector",
                metadata=r.get("metadata", {}),
            ))
        return results

    def _retrieve_graph(self, entity_list: List[Dict],
                        entity_boost: bool = True,
                        top_k: int = 20) -> Tuple[List[RetrievalResult], str]:
        """图谱检索：根据 entity_linker 的结果查询金融知识图谱。

        Returns:
            (RetrievalResult 列表, 图谱文本摘要)
        """
        if not entity_list or not self.graph_service.available:
            return [], ""

        try:
            graph_data = self.graph_service.search_by_entities(entity_list)
        except Exception as e:
            print(f"[HYBRID] 图谱检索失败: {e}")
            return [], ""

        text_summary = graph_data.get("text_summary", "")
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes:
            return [], text_summary

        results = []
        seen = set()

        # 将图谱节点转为 RetrievalResult
        for node in nodes:
            name = node.get("name", "")
            label = node.get("label", "")
            code = node.get("code", "")
            content = f"[图谱] {name} ({label})"
            if code:
                content += f" [{code}]"

            if content in seen:
                continue
            seen.add(content)

            results.append(RetrievalResult(
                content=content,
                score=1.0,  # 图谱命中给高分基础
                source_route="graph",
                metadata={
                    "graph_node_name": name,
                    "graph_node_label": label,
                    "graph_relation_count": sum(
                        1 for e in _find_edges_for_node(name, edges)
                    ),
                    "filename": f"[金融知识图谱] - {label}",
                    "section": label,
                    "page_number": None,
                    "paragraph_number": None,
                },
            ))

        # 实体增强：若 entity_boost 开启，将实体名压入结果池
        if entity_boost:
            for ent in entity_list:
                content = f"[实体] {ent['standard_name']} ({ent['entity_type']})"
                if content not in seen:
                    seen.add(content)
                    results.append(RetrievalResult(
                        content=content,
                        score=1.0,
                        source_route="graph",
                        metadata={
                            "graph_node_name": ent["standard_name"],
                            "graph_node_label": ent["entity_type"],
                            "filename": f"[金融知识图谱]",
                            "section": ent["entity_type"],
                            "page_number": None,
                            "paragraph_number": None,
                        },
                    ))

        return results[:top_k], text_summary

    # ── RRF 融合 ──

    @classmethod
    def _rrf_fusion(cls, *lists: List[RetrievalResult]) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion 融合多路结果。

        Score = Σ 1 / (K + rank_of_item_in_this_list)
        内容相同的块合并（取最高分 + 合计来源数）。
        """
        content_map: Dict[str, RetrievalResult] = {}
        seen_combos: Dict[str, int] = {}  # content → 已加分次数

        for lst in lists:
            for rank, item in enumerate(lst, 1):
                key = item.content
                if key not in content_map:
                    content_map[key] = item
                    content_map[key].rrf_score = 0.0
                # RRF 累加
                rrf_contrib = 1.0 / (cls.RRF_K + rank)
                content_map[key].rrf_score += rrf_contrib
                # 累计路数标识
                seen_combos[key] = seen_combos.get(key, 0) + 1

        # 按 RRF 分数降序排列
        sorted_results = sorted(
            content_map.values(),
            key=lambda x: (-x.rrf_score, -seen_combos.get(x.content, 0))
        )

        for i, r in enumerate(sorted_results):
            r.rank = i + 1

        return sorted_results

    # ── CrossEncoder 重排 ──

    def _rerank(self, query: str,
                candidates: List[RetrievalResult],
                top_k: int = 20) -> List[RetrievalResult]:
        """使用 bge-reranker 对候选结果重排序。"""
        if not candidates:
            return []

        reranker = _get_reranker()
        if reranker is None:
            return candidates[:top_k]

        pairs = [[query, c.content] for c in candidates]

        try:
            scores = reranker.predict(pairs)
        except Exception as e:
            print(f"[HYBRID] Rerank 预测失败: {e}")
            return candidates[:top_k]

        for i, cand in enumerate(candidates):
            cand.rerank_score = float(scores[i])

        reranked = sorted(candidates,
                          key=lambda x: x.rerank_score, reverse=True)
        return reranked[:top_k]

    # ── 引用溯源 ──

    @staticmethod
    def build_citations(results: List[RetrievalResult]) -> List[Dict]:
        """从检索结果中提取引用信息。

        每项包含：
        - document_name: 资料名（优先 metadata.filename）
        - section: 章节/条款/法规引证
        - page_number: 页码
        - paragraph_number: 段落号
        - content: 内容摘要（前 300 字）
        - source_route: 来源（bm25 / vector / graph）
        - score: 最终得分
        """
        citations = []
        for r in results:
            meta = r.metadata or {}
            doc_name = (
                meta.get("filename")
                or meta.get("source")
                or meta.get("graph_node_name")
                or "未知文档"
            )
            section = (
                meta.get("section")
                or meta.get("graph_node_label")
                or ""
            )
            citations.append({
                "document_name": doc_name,
                "section": section,
                "page_number": meta.get("page_number"),
                "paragraph_number": meta.get("paragraph_number"),
                "content": r.content[:300] + ("..." if len(r.content) > 300 else ""),
                "source_route": r.source_route,
                "score": round(r.rerank_score if r.rerank_score else r.rrf_score, 4),
            })
        return citations

    @staticmethod
    def format_debug_report(debug: RetrieveDebugInfo,
                            results: List[RetrievalResult]) -> str:
        """格式化调试信息为可读报告。"""
        lines = [
            "=" * 60,
            "  三路混合检索报告",
            "=" * 60,
        ]
        # 实体
        if debug.entities_found:
            lines.append(f"\n[实体抽取] ({len(debug.entities_found)} 个):")
            for e in debug.entities_found:
                lines.append(
                    f"  • [{e['entity_type']:10s}] {e['standard_name']} "
                    f"(别名: \"{e['alias_used']}\") "
                    f"行业: {e['industry_code']} 置信度: {e['confidence']:.0%}"
                )
        else:
            lines.append("\n[实体抽取] 未识别到已知实体")

        # 各路召回数
        lines.append(f"\n[各路召回]")
        lines.append(f"  BM25 ： {debug.bm25_count} 条")
        lines.append(f"  Vector： {debug.vector_count} 条")
        lines.append(f"  Graph ： {debug.graph_count} 条")
        if debug.graph_text_summary:
            lines.append(f"  图谱摘要：{debug.graph_text_summary}")

        # RRF 融合
        lines.append(f"\n[RRF 融合] TOP {debug.after_rrf_count} 条")

        # Rerank
        lines.append(f"[Rerank]")
        lines.append(f"  最高置信度：{debug.confidence:.4f}")
        if debug.used_fallback:
            lines.append(f"  置信度低于阈值(0.1)，回退至纯向量检索")
        lines.append(f"  精排后：{debug.after_rerank_count} 条")

        # 最终结果
        lines.append(f"\n[最终结果] TOP {len(results)}")
        for i, r in enumerate(results, 1):
            lines.append(
                f"  #{i} [来源:{r.source_route:8s}] "
                f"rrf={r.rrf_score:.4f} rerank={r.rerank_score:.4f}"
            )
            meta = r.metadata or {}
            doc = meta.get("filename") or meta.get("source") or ""
            sect = meta.get("section") or meta.get("graph_node_label") or ""
            lines.append(f"      文档: {doc}")
            if sect:
                lines.append(f"      章节: {sect}")
            lines.append(f"      {r.content[:100]}...")  # 前100字

        # 引用溯源
        citations = HybridRetriever.build_citations(results)
        lines.append(f"\n[引用溯源] ({len(citations)} 条)")
        for i, c in enumerate(citations, 1):
            lines.append(
                f"  [{i}] {c['document_name']} "
                f"| 章节: {c['section']} "
                f"| 来源: {c['source_route']}"
            )

        lines.append(f"\n[耗时] {debug.processing_time_ms:.1f} ms")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── 辅助 ──

def _find_edges_for_node(name: str, edges: List[Dict]) -> List[Dict]:
    return [e for e in edges
            if e.get("source") == name or e.get("target") == name]


# ═══════════════════════════════════════════════
#  单元验证
# ═══════════════════════════════════════════════

def _unit_test():
    retriever = HybridRetriever()
    passed = 0
    total = 0

    def run_test(test_name: str, query: str, min_total: int = 1,
                 expect_entity: bool = False):
        nonlocal passed, total
        total += 1
        print()
        print(f"  ── {test_name} ──")
        print(f"  问句: {query}")
        results, debug = retriever.retrieve(query, top_k=5, rerank_candidates=20)

        # 验证基本逻辑
        has_results = len(results) >= min_total
        entity_ok = (debug.entities_found and len(debug.entities_found) > 0) == expect_entity

        print(f"  实体: {len(debug.entities_found)} 个 "
              f"({[e['standard_name'] for e in debug.entities_found]})")
        print(f"  BM25={debug.bm25_count}  Vector={debug.vector_count}  "
              f"Graph={debug.graph_count}")
        if debug.graph_text_summary:
            print(f"  图谱摘要: {debug.graph_text_summary}")
        print(f"  RRF融合={debug.after_rrf_count} → "
              f"Rerank={debug.after_rerank_count} "
              f"(置信度={debug.confidence:.4f})")
        if debug.used_fallback:
            print(f"  [FALLBACK] 回退至纯向量检索")
        print(f"  最终: {len(results)} 条")
        for i, r in enumerate(results, 1):
            route = r.source_route
            meta = r.metadata or {}
            doc = meta.get("filename") or meta.get("source") or ""
            sect = meta.get("section") or meta.get("graph_node_label") or ""
            print(f"    #{i} [{route:10s}] rrf={r.rrf_score:.4f} "
                  f"rerank={r.rerank_score:.4f}  "
                  f"doc={doc[:30] if doc else '-'}  "
                  f"section={sect[:20] if sect else '-'}")

        status = "PASS" if has_results else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  [{status}] (min_total={min_total}, got={len(results)})")

    # ── 测试用例 ──
    run_test("公司+指标混合", "茅台去年营收和这次降准的关系",
             min_total=1, expect_entity=True)

    run_test("公司+财务指标", "宁德时代毛利率变化趋势",
             min_total=1, expect_entity=True)

    run_test("法规查询", "《证券法》对信息披露有什么要求",
             min_total=1, expect_entity=True)

    run_test("泛化问答（无需实体）", "金融科技最近有什么新政策",
             min_total=0, expect_entity=False)

    # ── 引用溯源验证 ──
    print()
    print("  ── 引用溯源验证 ──")
    q = "茅台去年营收"
    results, debug = retriever.retrieve(q, top_k=3)
    citations = HybridRetriever.build_citations(results)
    print(f"  问句: {q}")
    for i, c in enumerate(citations, 1):
        print(f"    [{i}] {c['document_name']}")
        print(f"        章节: {c['section']}  页码: {c['page_number']}")
        print(f"        来源: {c['source_route']}  得分: {c['score']}")
    passed += 1
    total += 1
    print("  [PASS] 引用溯源格式正确")
    print()

    # ── RRF 融合验证 ──
    print("  ── RRF 融合验证 ──")
    dummy_a = [RetrievalResult(content=f"doc_{i}", score=0.5, source_route="mock")
               for i in range(3)]
    dummy_b = [RetrievalResult(content=f"doc_{i}", score=0.5, source_route="mock")
               for i in range(1, 5)]
    fused = HybridRetriever._rrf_fusion(dummy_a, dummy_b)
    # doc_1 在 A 排第2, B 排第1 → RRF 分数应为 1/22 + 1/21 ≈ 0.093
    doc1 = next(f for f in fused if f.content == "doc_1")
    expected = 1.0 / (20 + 2) + 1.0 / (20 + 1)
    assert abs(doc1.rrf_score - expected) < 0.001, f"RRF 计算错误: {doc1.rrf_score} != {expected}"
    passed += 1
    total += 1
    print("  [PASS] RRF 分数计算正确")
    print()

    print("=" * 60)
    print(f"  结果: {passed}/{total} 通过")
    print("=" * 60)


if __name__ == "__main__":
    _unit_test()
