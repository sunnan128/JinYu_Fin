# ── 杏林AI 医学文档解析与切分 ──
# 决策记录：
# - 复用 LexAI 的 DocumentChunk 结构，保持与向量库 / API 接口一致
# - 临床指南 PDF：按章节标题正则切（如 "1 概述" / "2.3.1 治疗原则"）
# - 药品说明书：按【项】二级标题切（【药品名称】【适应症】【用法用量】...）
# - MIN_CHUNK_SIZE=120：医学段落更长，避免过碎（法律版为 80）
# - 未识别结构时回退到段落切分，保证不丢文本
#
# 用法（在 qa_service / 上传链路中把 DocumentParser 换成 MedicalDocumentParser 即可）：
#   from utils.medical_document_parser import MedicalDocumentParser
#   chunks = MedicalDocumentParser.parse_file(path)

import os
import re
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# ── 复用基础结构 ──
try:
    from .document_parser import DocumentChunk
except ImportError:
    try:
        from document_parser import DocumentChunk
    except ImportError:
        class DocumentChunk:
            def __init__(self, content: str, page_number=None,
                         paragraph_number=None, metadata=None):
                self.id = str(uuid.uuid4())
                self.content = content
                self.page_number = page_number
                self.paragraph_number = paragraph_number
                self.metadata = metadata or {}


class MedicalDocumentParser:
    # 医学段落普遍更长，提高最小合并阈值
    MIN_CHUNK_SIZE = 120

    # 药品说明书常见二级标题（顺序无关，用于【项】切分）
    DRUG_LABEL_SECTIONS = [
        "药品名称", "成份", "性状", "适应症", "功能主治", "规格",
        "用法用量", "不良反应", "禁忌", "注意事项", "孕妇及哺乳期妇女用药",
        "儿童用药", "老年用药", "药物相互作用", "药物过量", "药理毒理",
        "药代动力学", "贮藏", "包装", "有效期", "批准文号", "生产企业",
    ]

    # 指南章节标题：1 / 1.2 / 2.3.1 后接中文（整行标题）
    GUIDELINE_HEADING = re.compile(r'^[ \t]*(\d+(?:\.\d+)*)[ \t]+[一二三四五六七八九十百千\d]*[一-龥]+')

    # 药品说明书【项】标题
    DRUG_LABEL_HEADING = re.compile(r'^[ \t]*(【[^】]{1,12}】)')

    @staticmethod
    def parse_pdf(file_path: str) -> List[DocumentChunk]:
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is not installed. pip install pymupdf")
        chunks = []
        doc = fitz.open(file_path)
        filename = os.path.basename(file_path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if not text.strip():
                    continue
                page_chunks = MedicalDocumentParser._split_medical_text(
                    text, page_num + 1, filename
                )
                chunks.extend(page_chunks)
        finally:
            doc.close()
        return chunks

    @staticmethod
    def parse_word(file_path: str) -> List[DocumentChunk]:
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx is not installed. pip install python-docx")
        chunks = []
        doc = Document(file_path)
        filename = os.path.basename(file_path)
        current_page = 1
        paragraph_num = 0
        current_text = ""
        for para in doc.paragraphs:
            paragraph_num += 1
            text = para.text.strip()
            if not text:
                continue
            if MedicalDocumentParser._is_section_start(text) and current_text:
                chunks.append(DocumentChunk(
                    content=current_text.strip(),
                    page_number=current_page,
                    paragraph_number=paragraph_num - 1,
                    metadata={"filename": filename},
                ))
                current_text = text + "\n"
            else:
                current_text += text + "\n"
        if current_text.strip():
            chunks.append(DocumentChunk(
                content=current_text.strip(),
                page_number=current_page,
                paragraph_number=paragraph_num,
                metadata={"filename": filename},
            ))
        return chunks

    @staticmethod
    def _is_section_start(text: str) -> bool:
        if MedicalDocumentParser.DRUG_LABEL_HEADING.match(text):
            return True
        if MedicalDocumentParser.GUIDELINE_HEADING.match(text):
            return True
        return False

    @staticmethod
    def _split_medical_text(text: str, page_num: int, filename: str) -> List[DocumentChunk]:
        """优先按药品说明书【项】切，其次按指南章节标题切，最后段落回退。"""
        # 1) 药品说明书【项】结构检测
        if "【" in text and "】" in text:
            label_parts = re.split(r'(【[^】]{1,12}】)', text)
            has_sections = sum(
                1 for p in label_parts if MedicalDocumentParser.DRUG_LABEL_HEADING.match(p.strip())
            )
            if has_sections >= 2:
                return MedicalDocumentParser._split_by_label(label_parts, page_num, filename)

        # 2) 指南章节标题结构检测
        guide_parts = re.split(r'(^[ \t]*\d+(?:\.\d+)*[ \t]+[一二三四五六七八九十百千\d]*[一-龥]+)',
                               text, flags=re.MULTILINE)
        has_headings = sum(
            1 for p in guide_parts if MedicalDocumentParser.GUIDELINE_HEADING.match(p.strip())
        )
        if has_headings >= 2:
            return MedicalDocumentParser._merge_and_wrap(guide_parts, page_num, filename)

        # 3) 回退：按段落切分（保留文本不丢失）
        paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        return MedicalDocumentParser._merge_and_wrap(paras, page_num, filename, is_paragraph=True)

    @staticmethod
    def _split_by_label(label_parts: List[str], page_num: int, filename: str) -> List[DocumentChunk]:
        """label_parts 已被 (【项】) 捕获分组，每个【项】独立成块（不跨节合并）。"""
        chunks = []
        current_section = "正文"
        current_content = ""

        def flush():
            nonlocal current_content, current_section
            if current_content.strip():
                chunks.append({"section": current_section, "content": current_content.strip()})
            current_content = ""

        for part in label_parts:
            if not part:
                continue
            if MedicalDocumentParser.DRUG_LABEL_HEADING.match(part.strip()):
                flush()
                current_section = part.strip()
            else:
                current_content += part + "\n"
        flush()

        return [
            DocumentChunk(
                content=(f"{c['section']}\n{c['content']}") if c["section"] != "正文" else c["content"],
                page_number=page_num,
                paragraph_number=idx + 1,
                metadata={"filename": filename, "section": c["section"]},
            )
            for idx, c in enumerate(chunks)
        ]

    @staticmethod
    def _merge_and_wrap(parts: List[str], page_num: int, filename: str,
                        is_paragraph: bool = False) -> List[DocumentChunk]:
        """指南：标题与正文合并为每块，每个章节独立成块（医疗检索需细粒度）。
        极短块(<40字)并入下一块；纯段落回退走 MIN_CHUNK_SIZE 合并。"""
        if is_paragraph:
            raw = [{"content": p} for p in parts]
            return MedicalDocumentParser._wrap(raw, page_num, filename)

        raw = []
        buffer = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if MedicalDocumentParser.GUIDELINE_HEADING.match(part):
                if buffer:
                    raw.append(buffer)
                buffer = part + " "
            else:
                buffer += part + " "
        if buffer.strip():
            raw.append(buffer)

        merged = []
        for content in raw:
            content = content.strip()
            if not content:
                continue
            # 仅合并「几乎无正文」的纯标题块（<8字），正常小节独立成块
            if merged and len(merged[-1]) < 8:
                merged[-1] = merged[-1] + "\n" + content
            else:
                merged.append(content)

        return [
            DocumentChunk(
                content=m,
                page_number=page_num,
                paragraph_number=idx + 1,
                metadata={"filename": filename},
            )
            for idx, m in enumerate(merged)
        ]

    @staticmethod
    def _wrap(raw_chunks: List[Dict], page_num: int, filename: str) -> List[DocumentChunk]:
        """复用法律版的最小阈值合并逻辑（MIN_CHUNK_SIZE=120）。"""
        merged = []
        buffer = ""
        for ch in raw_chunks:
            content = ch["content"].strip()
            if not content:
                continue
            if len(buffer) < MedicalDocumentParser.MIN_CHUNK_SIZE:
                buffer = (buffer + "\n" + content).strip() if buffer else content
            else:
                merged.append(buffer)
                buffer = content
        if buffer:
            if merged and len(buffer) < MedicalDocumentParser.MIN_CHUNK_SIZE:
                merged[-1] = merged[-1] + "\n" + buffer
            else:
                merged.append(buffer)

        if not merged:
            return [DocumentChunk(content="", page_number=page_num, metadata={"filename": filename})]

        return [
            DocumentChunk(
                content=m,
                page_number=page_num,
                paragraph_number=idx + 1,
                metadata={"filename": filename},
            )
            for idx, m in enumerate(merged)
        ]

    @staticmethod
    def parse_file(file_path: str) -> List[DocumentChunk]:
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return MedicalDocumentParser.parse_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return MedicalDocumentParser.parse_word(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
