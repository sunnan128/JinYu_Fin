# -*- coding: utf-8 -*-
"""pytest 配置：将 backend/utils 加入导入路径，使测试可 import document_parser / hallucination_guard。"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "utils"))
sys.path.append(_ROOT)  # 项目根，支持 backend.* 包导入（接线集成测试需要）
