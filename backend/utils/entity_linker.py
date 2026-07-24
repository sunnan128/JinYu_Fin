# ── 金语AI 金融实体链接器 ──
# 维护金融实体别名表，从问句抽取实体并映射标准名+行业编码
# 仅规则+轻量匹配，不依赖 ML 模型
#
# 三层匹配：
#   1. 精确匹配 → alias 完全匹配问句子串
#   2. 同义匹配 → 已知同义词表（如 茅台→贵州茅台）
#   3. 上下文匹配 → 财务指标短语（如 营收→营业收入）
#
# 输出 entity_list: [{standard_name, entity_type, industry_code,
#                      confidence, alias_used}]

import re
from typing import List, Dict, Optional, Tuple

# ═══════════════════════════════════════════════════
#  金融实体别名表
#  ─ 结构：{标准名: {info}}
#  ─ entity_type: company | product | regulation | concept | indicator
#  ─ industry_code: 申万行业分类编码
# ═══════════════════════════════════════════════════

FINANCIAL_ENTITIES: Dict[str, Dict] = {
    # ── 白酒/食品 ──
    "贵州茅台": {
        "aliases": ["茅台", "贵州茅台", "飞天茅台", "53度飞天茅台"],
        "type": "company",
        "industry_code": "S24",        # 食品饮料
        "industry_name": "食品饮料",
        "stock_code": "600519.SH",
    },
    "五粮液": {
        "aliases": ["五粮液", "普五", "第八代五粮液"],
        "type": "company",
        "industry_code": "S24",
        "industry_name": "食品饮料",
        "stock_code": "000858.SZ",
    },
    "泸州老窖": {
        "aliases": ["泸州老窖", "国窖1573", "老窖"],
        "type": "company",
        "industry_code": "S24",
        "industry_name": "食品饮料",
        "stock_code": "000568.SZ",
    },
    "伊利股份": {
        "aliases": ["伊利", "伊利股份", "伊利牛奶"],
        "type": "company",
        "industry_code": "S24",
        "industry_name": "食品饮料",
        "stock_code": "600887.SH",
    },
    "海天味业": {
        "aliases": ["海天", "海天味业", "海天酱油"],
        "type": "company",
        "industry_code": "S24",
        "industry_name": "食品饮料",
        "stock_code": "603288.SH",
    },

    # ── 金融/银行 ──
    "工商银行": {
        "aliases": ["工商银行", "工行", "工商银行股份有限公司"],
        "type": "company",
        "industry_code": "A28",
        "industry_name": "银行",
        "stock_code": "601398.SH",
    },
    "招商银行": {
        "aliases": ["招商银行", "招行", "招商银行股份有限公司"],
        "type": "company",
        "industry_code": "A28",
        "industry_name": "银行",
        "stock_code": "600036.SH",
    },
    "中国平安": {
        "aliases": ["中国平安", "平安", "平安保险", "平安集团"],
        "type": "company",
        "industry_code": "A29",
        "industry_name": "保险",
        "stock_code": "601318.SH",
    },
    "中信证券": {
        "aliases": ["中信证券", "中信", "中信证券股份有限公司"],
        "type": "company",
        "industry_code": "A30",
        "industry_name": "证券",
        "stock_code": "600030.SH",
    },

    # ── 科技 ──
    "腾讯控股": {
        "aliases": ["腾讯", "腾讯控股", "腾讯公司", "TX"],
        "type": "company",
        "industry_code": "S22",
        "industry_name": "传媒",
        "stock_code": "0700.HK",
    },
    "阿里巴巴": {
        "aliases": ["阿里巴巴", "阿里", "阿里巴巴集团", "BABA"],
        "type": "company",
        "industry_code": "S22",
        "industry_name": "传媒",
        "stock_code": "9988.HK",
    },
    "宁德时代": {
        "aliases": ["宁德时代", "宁德", "CATL"],
        "type": "company",
        "industry_code": "S31",
        "industry_name": "电力设备",
        "stock_code": "300750.SZ",
    },
    "比亚迪": {
        "aliases": ["比亚迪", "BYD"],
        "type": "company",
        "industry_code": "S32",
        "industry_name": "汽车",
        "stock_code": "002594.SZ",
    },
    "华为": {
        "aliases": ["华为", "华为技术", "华为公司"],
        "type": "company",
        "industry_code": "S11",
        "industry_name": "电子",
    },

    # ── 能源 ──
    "中国石油": {
        "aliases": ["中国石油", "中石油", "中石油股份"],
        "type": "company",
        "industry_code": "S15",
        "industry_name": "石油石化",
        "stock_code": "601857.SH",
    },
    "中国石化": {
        "aliases": ["中国石化", "中石化", "中国石化股份"],
        "type": "company",
        "industry_code": "S15",
        "industry_name": "石油石化",
        "stock_code": "600028.SH",
    },
    "长江电力": {
        "aliases": ["长江电力"],
        "type": "company",
        "industry_code": "S17",
        "industry_name": "公用事业",
        "stock_code": "600900.SH",
    },

    # ── 金融产品 ──
    "沪深300ETF": {
        "aliases": ["沪深300ETF", "300ETF", "沪深300指数基金"],
        "type": "product",
        "industry_code": "A30",
        "industry_name": "基金",
    },
    "中证500ETF": {
        "aliases": ["中证500ETF", "500ETF", "中证500指数基金"],
        "type": "product",
        "industry_code": "A30",
        "industry_name": "基金",
    },
    "余额宝": {
        "aliases": ["余额宝", "天弘余额宝"],
        "type": "product",
        "industry_code": "A28",
        "industry_name": "货币基金",
    },

    # ── 法律法规 ──
    "证券法": {
        "aliases": ["证券法", "《证券法》", "中华人民共和国证券法"],
        "type": "regulation",
        "industry_code": "REG",
        "industry_name": "法律法规",
    },
    "基金法": {
        "aliases": ["基金法", "《基金法》", "中华人民共和国证券投资基金法"],
        "type": "regulation",
        "industry_code": "REG",
        "industry_name": "法律法规",
    },
    "公司法": {
        "aliases": ["公司法", "《公司法》", "中华人民共和国公司法"],
        "type": "regulation",
        "industry_code": "REG",
        "industry_name": "法律法规",
    },
    "上市公司信息披露管理办法": {
        "aliases": ["信息披露管理办法", "《信息披露管理办法》", "上市公司信息披露管理办法"],
        "type": "regulation",
        "industry_code": "REG",
        "industry_name": "法律法规",
    },

    # ── 金融概念 ──
    "下调存款准备金率": {
        "aliases": ["降准", "下调存款准备金率", "降低存款准备金率", "降准政策"],
        "type": "concept",
        "industry_code": "CON",
        "industry_name": "货币政策",
    },
    "贷款市场报价利率": {
        "aliases": ["LPR", "贷款市场报价利率", "基准利率"],
        "type": "concept",
        "industry_code": "CON",
        "industry_name": "利率",
    },
    "居民消费价格指数": {
        "aliases": ["CPI", "居民消费价格指数", "消费者物价指数", "通胀"],
        "type": "concept",
        "industry_code": "CON",
        "industry_name": "宏观经济",
    },
    "国内生产总值": {
        "aliases": ["GDP", "国内生产总值", "国民生产总值"],
        "type": "concept",
        "industry_code": "CON",
        "industry_name": "宏观经济",
    },
    "注册制": {
        "aliases": ["注册制", "股票发行注册制"],
        "type": "concept",
        "industry_code": "CON",
        "industry_name": "资本市场改革",
    },
    "北向资金": {
        "aliases": ["北向资金", "北上资金", "外资流入"],
        "type": "concept",
        "industry_code": "CON",
        "industry_name": "资金流向",
    },

    # ── 财务指标 ──
    "营业收入": {
        "aliases": ["营收", "营业收入", "主营业务收入", "总收入"],
        "type": "indicator",
        "industry_code": "FIN",
        "industry_name": "财务指标",
    },
    "净利润": {
        "aliases": ["净利润", "净利", "归母净利润", "扣非净利润"],
        "type": "indicator",
        "industry_code": "FIN",
        "industry_name": "财务指标",
    },
    "每股收益": {
        "aliases": ["每股收益", "EPS", "基本每股收益", "稀释每股收益"],
        "type": "indicator",
        "industry_code": "FIN",
        "industry_name": "财务指标",
    },
    "净资产收益率": {
        "aliases": ["净资产收益率", "ROE", "股东权益报酬率"],
        "type": "indicator",
        "industry_code": "FIN",
        "industry_name": "财务指标",
    },
    "毛利率": {
        "aliases": ["毛利率", "销售毛利率"],
        "type": "indicator",
        "industry_code": "FIN",
        "industry_name": "财务指标",
    },
    "资产负债率": {
        "aliases": ["资产负债率", "负债率", "杠杆率"],
        "type": "indicator",
        "industry_code": "FIN",
        "industry_name": "财务指标",
    },
}


# ═══════════════════════════════════════════════════
#  构建反向索引：alias → [标准名]
# ═══════════════════════════════════════════════════

def _build_alias_index() -> Dict[str, List[Tuple[str, float]]]:
    """构建别名 -> [(标准名, 置信度)] 映射。
    长别名（完全匹配）置信度更高，短别名可能有歧义。"""
    idx: Dict[str, List[Tuple[str, float]]] = {}
    for std_name, info in FINANCIAL_ENTITIES.items():
        for alias in info["aliases"]:
            # 完全匹配原词的置信度 = 1.0
            # 缩写/简写按长度折扣
            confidence = min(1.0, 0.6 + len(alias) * 0.02)
            idx.setdefault(alias, []).append((std_name, round(confidence, 2)))
            # 对纯中文别名同时建无空格索引（防问句中误加空格）
            if re.fullmatch(r'[\u4e00-\u9fff]+', alias):
                idx.setdefault(alias.replace(" ", ""), []).append(
                    (std_name, round(confidence - 0.05, 2))
                )
    return idx


ALIAS_INDEX = _build_alias_index()

# 按别名长度降序排列，优先匹配长词（防止"阿里"被"阿里巴巴"先匹配到）
for alias in ALIAS_INDEX:
    ALIAS_INDEX[alias].sort(key=lambda x: -len(x[0]))


class EntityLinker:
    """金融实体链接器：从问句抽取实体 → 标准名 + 行业编码。"""

    def __init__(self, custom_entities: Optional[Dict] = None):
        """可选注入自定义实体表（图谱节点补充）。"""
        self._entities = {**FINANCIAL_ENTITIES}
        if custom_entities:
            self._entities.update(custom_entities)
            self._rebuild_index()

    def _rebuild_index(self):
        self._idx = _build_alias_index()

    # ── 公开 API ──

    def extract_entities(self, question: str) -> List[Dict]:
        """从问句中抽取实体，返回去重后的 entity_list。

        返回格式:
            [{
                "standard_name": str,    # 标准名称
                "entity_type": str,      # company|product|regulation|concept|indicator
                "industry_code": str,    # 行业编码
                "industry_name": str,    # 行业名称
                "confidence": float,     # 匹配置信度
                "alias_used": str,       # 问句中实际匹配到的别名
                "stock_code": str|None,  # 股票代码
            }]
        """
        results: Dict[str, Dict] = {}  # standard_name → result
        seen_positions: List[Tuple[int, int]] = []  # 记录已匹配的区间，避免重复

        # ── 第1步：长优先精确匹配 ──
        # 按别名长度降序排列匹配，确保"上市公司信息披露管理办法"
        # 优先于"证券法"被匹配
        all_aliases = sorted(
            ALIAS_INDEX.items(),
            key=lambda x: -len(x[0])
        )
        for alias, candidates in all_aliases:
            for match in re.finditer(re.escape(alias), question):
                start, end = match.start(), match.end()
                # 跳过已占用的区间
                if self._is_overlapping(start, end, seen_positions):
                    continue
                seen_positions.append((start, end))
                # 取置信度最高的候选
                best = max(candidates, key=lambda x: x[1])
                std_name, confidence = best
                entity = self._entities.get(std_name, {})
                if std_name not in results or results[std_name]["confidence"] < confidence:
                    results[std_name] = {
                        "standard_name": std_name,
                        "entity_type": entity.get("type", "unknown"),
                        "industry_code": entity.get("industry_code", ""),
                        "industry_name": entity.get("industry_name", ""),
                        "confidence": confidence,
                        "alias_used": alias,
                        "stock_code": entity.get("stock_code"),
                    }

        # ── 第2步：上下文短语匹配（财务指标短语） ──
        # 如"去年营收"中的"营收" → 营业收λ
        indicator_ctx = self._match_indicator_context(question, seen_positions)
        for ctx in indicator_ctx:
            std_name = ctx["standard_name"]
            if std_name not in results:
                results[std_name] = ctx

        # ── 排序输出 ──
        return sorted(
            results.values(),
            key=lambda x: (-x["confidence"], x["standard_name"])
        )

    # ── 内部逻辑 ──

    def _is_overlapping(self, start: int, end: int,
                        seen: List[Tuple[int, int]]) -> bool:
        """判断 [start, end) 是否与已匹配区间重叠。"""
        for s, e in seen:
            if not (end <= s or start >= e):
                return True
        return False

    def _match_indicator_context(self, question: str,
                                 seen: List[Tuple[int, int]]) -> List[Dict]:
        """匹配财务指标上下文。
        例如"营收"出现在"去年营收"、"Q3营收"等场景中。
        此处仅对未占用的文本区间做短语检测。"""
        results: Dict[str, Dict] = {}
        indicator_aliases = {
            "营收": "营业收入",
            "净利": "净利润",
            "利润": "净利润",
            "EPS": "每股收益",
            "ROE": "净资产收益率",
            "负债率": "资产负债率",
            "毛利率": "毛利率",
        }
        for alias, std_name in indicator_aliases.items():
            for match in re.finditer(re.escape(alias), question):
                start, end = match.start(), match.end()
                if self._is_overlapping(start, end, seen):
                    continue
                entity = self._entities.get(std_name, {})
                results[std_name] = {
                    "standard_name": std_name,
                    "entity_type": entity.get("type", "indicator"),
                    "industry_code": entity.get("industry_code", "FIN"),
                    "industry_name": entity.get("industry_name", "财务指标"),
                    "confidence": 0.85,
                    "alias_used": alias,
                    "stock_code": None,
                }
        return list(results.values())

    def link_to_graph_query(self, entity_list: List[Dict]) -> Dict:
        """将实体列表转换为图谱检索所需的查询结构。
        供后续图数据库（Neo4j / NetworkX）使用。"""
        return {
            "entities": [
                {
                    "name": e["standard_name"],
                    "type": e["entity_type"],
                    "industry_code": e["industry_code"],
                    "stock_code": e.get("stock_code"),
                }
                for e in entity_list
            ],
            "query_hints": {
                "company": [e["standard_name"]
                            for e in entity_list
                            if e["entity_type"] == "company"],
                "concept": [e["standard_name"]
                            for e in entity_list
                            if e["entity_type"] == "concept"],
                "indicator": [e["standard_name"]
                              for e in entity_list
                              if e["entity_type"] == "indicator"],
            },
        }

    @staticmethod
    def format_entity_report(entity_list: List[Dict]) -> str:
        """将 entity_list 格式化为可读报告（调试/日志用）。"""
        lines = ["已识别实体："]
        for e in entity_list:
            code = f" ({e['stock_code']})" if e.get("stock_code") else ""
            lines.append(
                f"  [{e['entity_type']:12s}] {e['standard_name']}{code}\n"
                f"    ├─ 别名: \"{e['alias_used']}\"\n"
                f"    ├─ 行业: {e['industry_name']} ({e['industry_code']})\n"
                f"    └─ 置信度: {e['confidence']:.0%}"
            )
        if not entity_list:
            lines.append("  (未识别到已知实体)")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════
#  单元验证
# ═══════════════════════════════════════════════════
# 运行 python entity_linker.py 触发

def _unit_test():
    linker = EntityLinker()
    passed = 0
    total = 0

    def check(test_name: str, question: str,
              expected: List[str], min_count: int = 1):
        nonlocal passed, total
        total += 1
        result = linker.extract_entities(question)
        found_std = [e["standard_name"] for e in result]
        is_ok = all(exp in found_std for exp in expected) and len(result) >= min_count
        status = "PASS" if is_ok else "FAIL"
        if is_ok:
            passed += 1
        print(f"  [{status}] {test_name}")
        print(f"         问句: {question}")
        print(f"         期望: {expected}")
        print(f"         命中: {found_std}")
        print(f"         报告:")
        print(EntityLinker.format_entity_report(result).replace(
            "\n", "\n           "
        ))
        print()

    print("=" * 60)
    print("  金语AI 金融实体链接器 —— 单元验证")
    print("=" * 60)
    print()

    # ── 1. 公司别名匹配 ──
    check(
        "公司别名 · 茅台→贵州茅台",
        "茅台去年营收是多少？",
        ["贵州茅台", "营业收入"],
    )
    check(
        "公司别名 · 阿里→阿里巴巴",
        "阿里Q3业绩如何？",
        ["阿里巴巴"],
    )
    check(
        "公司别名 · 招行→招商银行",
        "招行的不良贷款率是多少？",
        ["招商银行"],
    )

    # ── 2. 金融概念匹配 ──
    check(
        "概念别名 · 降准→下调存款准备金率",
        "这次降准对股市有什么影响？",
        ["下调存款准备金率"],
    )
    check(
        "概念别名 · CPI→居民消费价格指数",
        "最新CPI数据公布了吗？",
        ["居民消费价格指数"],
    )
    check(
        "概念别名 · GDP→国内生产总值",
        "中国2024年GDP增速目标是多少？",
        ["国内生产总值"],
    )

    # ── 3. 法规匹配 ──
    check(
        "法规匹配 · 证券法",
        "《证券法》对信息披露有什么要求？",
        ["证券法"],
    )

    # ── 4. 财务指标匹配 ──
    check(
        "指标匹配 · ROE",
        "茅台的ROE连续三年是多少？",
        ["贵州茅台", "净资产收益率"],
    )
    check(
        "指标匹配 · 毛利率",
        "宁德时代毛利率变化趋势",
        ["宁德时代", "毛利率"],
    )

    # ── 5. 多实体混合 ──
    check(
        "多实体混合 · 茅台营收+降准",
        "茅台去年营收和这次降准的关系",
        ["贵州茅台", "营业收入", "下调存款准备金率"],
        min_count=3,
    )

    # ── 6. 图谱查询输出验证 ──
    q = "比亚迪的资产负债率"
    entity_list = linker.extract_entities(q)
    graph_query = linker.link_to_graph_query(entity_list)
    print("  [VERIFY] 图谱查询结构:")
    print(f"          entities: {[e['name'] for e in graph_query['entities']]}")
    print(f"          company hints: {graph_query['query_hints']['company']}")
    print(f"          indicator hints: {graph_query['query_hints']['indicator']}")
    assert any(e["name"] == "比亚迪" for e in graph_query["entities"])
    assert any(e["name"] == "资产负债率" for e in graph_query["entities"])
    passed += 1
    total += 1
    print("  [PASS] 图谱查询格式正确")
    print()

    # ── 7. 空问句 ──
    empty_result = linker.extract_entities("你好")
    assert len(empty_result) == 0
    passed += 1
    total += 1
    print("  [PASS] 空问句(无实体时返回空列表)")
    print()

    print("=" * 60)
    print(f"  结果: {passed}/{total} 通过")
    print("=" * 60)


if __name__ == "__main__":
    _unit_test()
