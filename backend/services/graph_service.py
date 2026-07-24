# ── 金融知识图谱检索服务（Neo4j） ──
# 配合 entity_linker 的 entity_list 跑 Cypher 查询
# 图结构：
#   (Company)-[:BELONGS_TO]->(Industry)
#   (Company)-[:ISSUES]->(Product)
#   (Regulation)-[:GOVERNS]->(Industry)
#   (Concept)  (独立节点)
#   (Company)-[:RELATED_TO]->(Concept)
#
# 注意：Neo4j 为可选依赖，未连接时 graph_search 返回空列表

import os
import re
import time
from typing import List, Dict, Any, Optional, Callable

from backend.config import settings


# ── 本地 mock 知识图谱（Neo4j 不可用时的降级） ──
# 包含常见金融实体的关系数据，供快速验证
_MOCK_GRAPH: Dict[str, List[Dict]] = {
    "nodes": [
        # 行业
        {"id": "ind_1", "label": "Industry",  "name": "食品饮料",      "code": "S24"},
        {"id": "ind_2", "label": "Industry",  "name": "银行",          "code": "A28"},
        {"id": "ind_3", "label": "Industry",  "name": "保险",          "code": "A29"},
        {"id": "ind_4", "label": "Industry",  "name": "证券",          "code": "A30"},
        {"id": "ind_5", "label": "Industry",  "name": "电力设备",      "code": "S31"},
        {"id": "ind_6", "label": "Industry",  "name": "汽车",          "code": "S32"},
        {"id": "ind_7", "label": "Industry",  "name": "传媒",          "code": "S22"},
        # 公司
        {"id": "comp_1", "label": "Company",  "name": "贵州茅台",       "code": "600519.SH"},
        {"id": "comp_2", "label": "Company",  "name": "五粮液",         "code": "000858.SZ"},
        {"id": "comp_3", "label": "Company",  "name": "工商银行",       "code": "601398.SH"},
        {"id": "comp_4", "label": "Company",  "name": "招商银行",       "code": "600036.SH"},
        {"id": "comp_5", "label": "Company",  "name": "中国平安",       "code": "601318.SH"},
        {"id": "comp_6", "label": "Company",  "name": "宁德时代",       "code": "300750.SZ"},
        {"id": "comp_7", "label": "Company",  "name": "比亚迪",         "code": "002594.SZ"},
        # 产品
        {"id": "prod_1", "label": "Product",  "name": "飞天茅台"},
        {"id": "prod_2", "label": "Product",  "name": "国窖1573"},
        {"id": "prod_3", "label": "Product",  "name": "余额宝"},
        # 法规
        {"id": "reg_1",  "label": "Regulation","name": "证券法"},
        {"id": "reg_2",  "label": "Regulation","name": "基金法"},
        {"id": "reg_3",  "label": "Regulation","name": "上市公司信息披露管理办法"},
        # 概念
        {"id": "conc_1","label": "Concept",   "name": "下调存款准备金率"},
        {"id": "conc_2","label": "Concept",   "name": "贷款市场报价利率"},
        {"id": "conc_3","label": "Concept",   "name": "居民消费价格指数"},
        {"id": "conc_4","label": "Concept",   "name": "注册制"},
    ],
    "edges": [
        ("comp_1", "ind_1", "BELONGS_TO"),
        ("comp_2", "ind_1", "BELONGS_TO"),
        ("comp_3", "ind_2", "BELONGS_TO"),
        ("comp_4", "ind_2", "BELONGS_TO"),
        ("comp_5", "ind_3", "BELONGS_TO"),
        ("comp_6", "ind_5", "BELONGS_TO"),
        ("comp_7", "ind_6", "BELONGS_TO"),
        ("comp_1", "prod_1", "ISSUES"),
        ("comp_2", "prod_2", "ISSUES"),
        ("comp_4", "prod_3", "ISSUES"),
        ("reg_1",  "ind_2", "GOVERNS"),
        ("reg_1",  "ind_3", "GOVERNS"),
        ("reg_1",  "ind_4", "GOVERNS"),
        ("reg_2",  "ind_2", "GOVERNS"),
        ("reg_3",  "ind_4", "GOVERNS"),
    ],
}

# 构建快速查找：node name → node
_MOCK_NODES_BY_NAME: Dict[str, Dict] = {}
for n in _MOCK_GRAPH["nodes"]:
    _MOCK_NODES_BY_NAME[n["name"]] = n

# 构建邻接表
_MOCK_ADJ: Dict[str, List[Dict]] = {}
for src, dst, rel in _MOCK_GRAPH["edges"]:
    src_node = next((n for n in _MOCK_GRAPH["nodes"] if n["id"] == src), None)
    dst_node = next((n for n in _MOCK_GRAPH["nodes"] if n["id"] == dst), None)
    if src_node and dst_node:
        _MOCK_ADJ.setdefault(src_node["name"], []).append({
            "relation": rel, "node_name": dst_node["name"],
            "node_label": dst_node["label"], "node": dst_node,
        })
    if dst_node and dst_node["name"]:
        _MOCK_ADJ.setdefault(dst_node["name"], []).append({
            "relation": rel, "node_name": src_node["name"],
            "node_label": src_node["label"], "node": src_node,
        })


class GraphService:
    """金融知识图谱检索服务。

    连接 Neo4j 并执行 Cypher 查询，若 Neo4j 不可用则降级到本地 mock 图谱。
    """

    def __init__(self):
        self._driver = None
        self._connected = False
        self._use_mock = False
        self._init_connection()

    def _init_connection(self):
        """尝试连接 Neo4j，失败后降级到 mock 图谱。"""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
            self._driver.verify_connectivity()
            self._connected = True
            print("[GRAPH] Neo4j 连接成功: {}".format(settings.NEO4J_URI))
        except Exception as e:
            self._use_mock = True
            print("[GRAPH] Neo4j 不可用 ({}), 降级到内置 mock 图谱".format(str(e)))

    @property
    def available(self) -> bool:
        return self._connected or self._use_mock

    # ── 公开 API ──

    def search_by_entities(self, entity_list: List[Dict]) -> Dict[str, Any]:
        """根据 entity_linker 输出的 entity_list 检索图谱。

        Args:
            entity_list: entity_linker 输出，每项含 standard_name, entity_type, industry_code

        Returns:
            {
                "nodes": [...],       # 匹配到的图谱节点
                "edges": [...],       # 关系描述
                "text_summary": str,  # 可读文本摘要（供 LLM 上下文中拼接）
            }
        """
        if not entity_list:
            return {"nodes": [], "edges": [], "text_summary": ""}

        # 收集所有需要查询的标准名
        company_names = [e["standard_name"] for e in entity_list
                         if e["entity_type"] == "company"]
        concept_names = [e["standard_name"] for e in entity_list
                         if e["entity_type"] in ("concept",)]
        indicator_names = [e["standard_name"] for e in entity_list
                           if e["entity_type"] == "indicator"]
        regulation_names = [e["standard_name"] for e in entity_list
                            if e["entity_type"] == "regulation"]
        product_names = [e["standard_name"] for e in entity_list
                         if e["entity_type"] == "product"]
        # 所有标准名（不限类型），兜底搜索
        all_names = [e["standard_name"] for e in entity_list]
        industry_codes = list(set(
            e["industry_code"] for e in entity_list if e.get("industry_code")
        ))

        if self._connected:
            return self._search_neo4j(
                company_names, concept_names, indicator_names, industry_codes,
                entity_list,
            )
        else:
            return self._search_mock(
                company_names, concept_names, indicator_names,
                regulation_names, product_names, all_names,
                industry_codes, entity_list,
            )

    def search_topics(self, keywords: List[str]) -> List[Dict]:
        """按关键词搜索图谱节点（模糊匹配名称）。"""
        results = []
        seen = set()
        for kw in keywords:
            for name, node in _MOCK_NODES_BY_NAME.items():
                if kw.lower() in name.lower() and name not in seen:
                    seen.add(name)
                    results.append({
                        "name": node["name"],
                        "label": node["label"],
                        "code": node.get("code", ""),
                    })
        return results

    # ── Neo4j 查询 ──

    def _search_neo4j(self, company_names, concept_names, indicator_names,
                      industry_codes, entity_list) -> Dict[str, Any]:
        """通过 Neo4j Cypher 查询。"""
        nodes = []
        edges = []
        with self._driver.session() as session:
            # 查询公司及其关联
            for name in company_names:
                result = session.run(
                    """MATCH (c:Company {name: $name})
                       OPTIONAL MATCH (c)-[:BELONGS_TO]->(i:Industry)
                       OPTIONAL MATCH (c)-[:ISSUES]->(p:Product)
                       OPTIONAL MATCH (c)-[:RELATED_TO]->(x:Concept)
                       RETURN c, i, p, x""",
                    name=name,
                )
                for record in result:
                    if record["c"]:
                        nodes.append(self._neo4j_node_to_dict(record["c"]))
                    if record["i"]:
                        nodes.append(self._neo4j_node_to_dict(record["i"]))
                    if record["p"]:
                        nodes.append(self._neo4j_node_to_dict(record["p"]))
                    if record["x"]:
                        nodes.append(self._neo4j_node_to_dict(record["x"]))

            # 查询概念
            for name in concept_names:
                result = session.run(
                    "MATCH (x:Concept {name: $name}) RETURN x", name=name
                )
                for record in result:
                    if record["x"]:
                        nodes.append(self._neo4j_node_to_dict(record["x"]))

        # 去重
        unique: Dict[str, Dict] = {}
        for n in nodes:
            key = "{}:{}".format(n.get("label", ""), n.get("name", ""))
            unique[key] = n
        text = GraphService._build_text_summary(list(unique.values()), edges)
        return {"nodes": list(unique.values()), "edges": edges, "text_summary": text}

    @staticmethod
    def _neo4j_node_to_dict(node) -> Dict:
        return dict(node)

    # ── Mock 查询 ──

    def _search_mock(self, company_names, concept_names, indicator_names,
                     regulation_names, product_names, all_names,
                     industry_codes, entity_list) -> Dict[str, Any]:
        """通过本地 mock 图谱查询。"""
        nodes = []
        seen_ids = set()

        def add_node(name: str):
            node = _MOCK_NODES_BY_NAME.get(name)
            if node and node["id"] not in seen_ids:
                seen_ids.add(node["id"])
                nodes.append(node)

        for name in company_names:
            add_node(name)
            for neighbor in _MOCK_ADJ.get(name, []):
                add_node(neighbor["node_name"])

        for name in concept_names:
            add_node(name)
            for neighbor in _MOCK_ADJ.get(name, []):
                add_node(neighbor["node_name"])

        for name in regulation_names:
            add_node(name)
            for neighbor in _MOCK_ADJ.get(name, []):
                add_node(neighbor["node_name"])

        for name in product_names:
            add_node(name)
            for neighbor in _MOCK_ADJ.get(name, []):
                add_node(neighbor["node_name"])

        # 兜底：尝试所有标准名（未匹配到的旧条目）
        for name in all_names:
            add_node(name)

        # 通过行业编码找行业节点
        for code in industry_codes:
            for n in _MOCK_GRAPH["nodes"]:
                if n.get("code") == code and n["id"] not in seen_ids:
                    seen_ids.add(n["id"])
                    nodes.append(n)

        text = GraphService._build_text_summary(nodes, _MOCK_GRAPH["edges"])
        edges_formatted = []
        for src_id, dst_id, rel in _MOCK_GRAPH["edges"]:
            src = next((n for n in _MOCK_GRAPH["nodes"] if n["id"] == src_id), None)
            dst = next((n for n in _MOCK_GRAPH["nodes"] if n["id"] == dst_id), None)
            if src and dst:
                edges_formatted.append({
                    "source": src["name"], "target": dst["name"],
                    "relation": rel,
                })
        return {"nodes": nodes, "edges": edges_formatted, "text_summary": text}

    # ── 公共工具 ──

    @staticmethod
    def _build_text_summary(nodes: List[Dict],
                            edges: List[Any]) -> str:
        """将图谱节点组装为自然文本段落，供 LLM 拼接上下文。"""
        parts = []
        companies = [n for n in nodes if n.get("label") == "Company"]
        industries = [n for n in nodes if n.get("label") == "Industry"]
        products = [n for n in nodes if n.get("label") == "Product"]
        concepts = [n for n in nodes if n.get("label") == "Concept"]
        regulations = [n for n in nodes if n.get("label") == "Regulation"]

        if companies:
            names = ["{} ({})".format(n["name"], n.get("code", ""))
                     for n in companies]
            parts.append("相关公司：" + "、".join(names))
        if industries:
            codes = ["{} ({})".format(n["name"], n.get("code", ""))
                     for n in industries]
            parts.append("所属行业：" + "、".join(codes))
        if products:
            parts.append("相关产品：" + "、".join(n["name"] for n in products))
        if concepts:
            parts.append("相关概念：" + "、".join(n["name"] for n in concepts))
        if regulations:
            parts.append("监管法规：" + "、".join(n["name"] for n in regulations))

        return "\n".join(parts)

    def close(self):
        if self._driver:
            self._driver.close()
