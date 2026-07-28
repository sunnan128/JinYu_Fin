# ── 文档解析与切分 ──
# 决策记录：
# - PDF 用 PyMuPDF(fitz) 解析，Word 用 python-docx
# - 按"第X条/款/章/节"正则切分，适合法律文书结构
# - MIN_CHUNK_SIZE=80：相邻小片段自动合并，减少向量编码次数，
#   首批上传时提速 3~5 倍（原策略每个条款独立成块，数量过多）

import os
import re
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Import optional dependencies with error handling
try:
    import fitz
except ImportError:
    fitz = None
    print("Warning: PyMuPDF (fitz) not installed. PDF parsing will not work.")

try:
    from docx import Document
except ImportError:
    Document = None
    print("Warning: python-docx not installed. Word document parsing will not work.")

def extract_front_matter(text: str):
    """【新增·仅解析元数据，不改任何切分逻辑】
    解析 Markdown 文件顶部"爬虫写入的 front-matter"——连续多行 '# 键: 值'
    形式（例：# 权威机构: central_bank / # 令号: 国务院令第768号 /
    # 排序键: 768 / # 施行日期: 2025-01-01 / # 效力状态: current）。
    返回 (cleaned_text, meta_dict)：
      - cleaned_text：剔除全部 front-matter 行后的正文（防止元数据污染 chunk）；
      - meta_dict：解析出的键值对，供后续写入 chunk.metadata（不会改动原有关键字段）。
    兼容性（仅新增、不改原有行为）：
      - 原代码只剔除 # 标题/来源/链接: 三行；本函数剔除顶部全部 '# 键: 值' 行，
        对这三类行的处理结果与原来一致，其余正文（如真正的 Markdown 章节标题
        '# 第一章' 因不含冒号）不会被误删。
      - 若文件顶部首行不是 '# 键: 值' 格式，则判定为无 front-matter，原样返回，
        原有 Markdown 标题/条款切分逻辑完全不受影响。
    """
    meta = {}
    lines = text.split('\n')
    idx = 0
    for line in lines:
        m = re.match(r'^#\s*([^:：\n]+)[：:]\s*(.*)$', line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            if key:
                meta[key] = val
                idx += 1
            else:
                break
        else:
            break
    if idx == 0:
        # 首行即非 front-matter → 整篇无 front-matter，原样返回，不影响 Markdown 章节标题切分
        return text, {}
    cleaned = '\n'.join(lines[idx:]).lstrip('\n')
    return cleaned, meta

class DocumentChunk:
    def __init__(self, content: str, page_number: Optional[int] = None, 
                 paragraph_number: Optional[int] = None, metadata: Dict[str, Any] = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.page_number = page_number
        self.paragraph_number = paragraph_number
        self.metadata = metadata or {}

class DocumentParser:
    # 最小合并阈值：少于该字数的相邻片段自动合并
    MIN_CHUNK_SIZE = 80

    @staticmethod
    def parse_pdf(file_path: str) -> List[DocumentChunk]:
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is not installed. Please install it with 'pip install pymupdf'.")
        
        chunks = []
        doc = fitz.open(file_path)
        filename = os.path.basename(file_path)
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if not text.strip():
                    continue
                    
                page_chunks = DocumentParser._split_legal_text(
                    text, 
                    page_num + 1, 
                    filename
                )
                chunks.extend(page_chunks)
        finally:
            doc.close()

        # 【B 方案】抽取 PDF 元数据并并入每个 chunk，使 L1 对 PDF 生效。
        # 与 .md front-matter 使用完全相同的键名，hallucination_guard 原样消费。
        # 异常时优雅降级：只保留 filename，不阻断解析/入库。
        try:
            from backend.utils.pdf_metadata_extractor import extract_pdf_metadata
            pdf_meta = extract_pdf_metadata(file_path)
        except Exception:
            pdf_meta = {}
        if pdf_meta:
            for ch in chunks:
                merged = dict(pdf_meta)
                merged.update(ch.metadata or {})
                merged["filename"] = filename  # 始终以实际文件名覆盖
                ch.metadata = merged

        return chunks
    
    @staticmethod
    def parse_word(file_path: str) -> List[DocumentChunk]:
        if Document is None:
            raise ImportError("python-docx is not installed. Please install it with 'pip install python-docx'.")
        
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
                
            if DocumentParser._is_legal_article_start(text) and current_text:
                chunks.append(DocumentChunk(
                    content=current_text.strip(),
                    page_number=current_page,
                    paragraph_number=paragraph_num - 1,
                    metadata={"filename": filename}
                ))
                current_text = text + "\n"
            else:
                current_text += text + "\n"
        
        if current_text.strip():
            chunks.append(DocumentChunk(
                content=current_text.strip(),
                page_number=current_page,
                paragraph_number=paragraph_num,
                metadata={"filename": filename}
            ))
        
        return chunks
    
    @staticmethod
    def parse_markdown(file_path: str) -> List[DocumentChunk]:
        """解析 Markdown 文件（.md / .markdown）。
        
        切分策略（按优先级）：
        1. Markdown 一级/二级标题（# / ##）→ 每个标题下内容独立成块
        2. 「第X条/章」法律条款结构 → 每条独立成块
        3. 空行分隔段落 → 段落级回退
        """
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as f:
                text = f.read()

        # 解析并剔除爬虫写入的 front-matter（兼容原逻辑：# 标题/来源/链接: 三种行
        # 仍被剔除；其余真正的 Markdown 章节标题因不含冒号不会被误删）。
        # front_meta 仅作为新增元数据并入 chunk，不改变原有切分与字段行为。
        text, front_meta = extract_front_matter(text)

        # 策略1：按 Markdown 标题切分
        heading_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
        all_heading_matches = list(heading_pattern.finditer(text))
        # 过滤掉元数据标题（# 标题: / # 来源: / # 链接:），只保留章节标题
        heading_matches = [
            m for m in all_heading_matches
            if not m.group(2).strip().startswith(("标题:", "来源:", "链接:"))
        ]
        
        chunks = []
        if len(heading_matches) >= 2:
            # 至少2个标题 → 按标题位置切分
            for i, match in enumerate(heading_matches):
                start = match.end()
                end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
                section_content = text[start:end].strip()
                if not section_content:
                    continue
                # section 名取自标题文字
                section_title = match.group(2).strip()
                full_content = f"{section_title}\n{section_content}"
                chunks.append(DocumentChunk(
                    content=full_content,
                    page_number=1,
                    paragraph_number=len(chunks) + 1,
                    metadata={
                        "filename": filename,
                        "section": section_title,
                        **front_meta,
                    }
                ))
            if chunks:
                return chunks

        # 策略2：按「第X条」法律条款切分
        # 关键修复：只在"条款起点"切分——前一个字符必须是句末标点/换行/全角空格/
        # 右括号等；刻意不含 、和 ，，从而避免把"本法第五十三条"这类行内引用
        # 误判为新的一条（否则一条法条会被切成多块）。
        article_pattern = re.compile(
            r'(?:^|(?<=[。；！？\r\n\t \u3000）】”》」]))'
            r'(第[零一二三四五六七八九十百千\d]+[条章节])'
        )
        parts = re.split(article_pattern, text)
        raw_sections = []
        buffer_section = "正文"
        buffer_content = ""
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if i % 2 == 1:  # 条款号
                if buffer_content.strip():
                    raw_sections.append((buffer_section, buffer_content))
                buffer_section = part
                buffer_content = ""
            else:
                buffer_content += part + " "
        if buffer_content.strip():
            raw_sections.append((buffer_section, buffer_content))

        if len(raw_sections) >= 2:
            for section_title, content in raw_sections:
                # 把条款号拼入内容正文，确保关键词检索能命中
                full_content = f"{section_title} {content.strip()}"
                chunks.append(DocumentChunk(
                    content=full_content,
                    page_number=1,
                    paragraph_number=len(chunks) + 1,
                    metadata={
                        "filename": filename,
                        "section": section_title,
                        **front_meta,
                    }
                ))
            return chunks

        # 策略3（回退）：按空行分段
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        for para in paragraphs:
            chunks.append(DocumentChunk(
                content=para,
                page_number=1,
                paragraph_number=len(chunks) + 1,
                metadata={"filename": filename, **front_meta}
            ))
        return chunks
    
    @staticmethod
    def _split_legal_text(text: str, page_num: int, filename: str) -> List[DocumentChunk]:
        chunks = []
        
        article_pattern = r'(?:第[零一二三四五六七八九十百千\d]+条|第[零一二三四五六七八九十百千\d]+款)'
        
        parts = re.split(f'({article_pattern})', text)
        
        if not parts:
            return [DocumentChunk(
                content=text.strip(),
                page_number=page_num,
                metadata={"filename": filename}
            )]
        
        current_content = ""
        para_num = 0
        
        for i in range(len(parts)):
            part = parts[i].strip()
            if not part:
                continue
                
            if re.match(article_pattern, part):
                if current_content:
                    chunks.append({
                        'content': current_content.strip(),
                        'para_num': para_num
                    })
                    para_num += 1
                current_content = part + " "
            else:
                current_content += part
        
        if current_content.strip():
            chunks.append({
                'content': current_content.strip(),
                'para_num': para_num
            })
        
        # Merge small consecutive chunks
        merged = []
        buffer = ""
        buffer_para = 0
        for ch in chunks:
            if len(buffer) < DocumentParser.MIN_CHUNK_SIZE:
                buffer += "\n" + ch['content'] if buffer else ch['content']
                if buffer_para == 0:
                    buffer_para = ch['para_num']
            else:
                merged.append({'content': buffer, 'para_num': buffer_para})
                buffer = ch['content']
                buffer_para = ch['para_num']
        if buffer:
            if merged and len(buffer) < DocumentParser.MIN_CHUNK_SIZE:
                merged[-1]['content'] += "\n" + buffer
            else:
                merged.append({'content': buffer, 'para_num': buffer_para})
        
        if not merged:
            return [DocumentChunk(
                content=text.strip(),
                page_number=page_num,
                metadata={"filename": filename}
            )]
        
        return [
            DocumentChunk(
                content=m['content'],
                page_number=page_num,
                paragraph_number=m['para_num'],
                metadata={"filename": filename}
            )
            for m in merged
        ]
    
    @staticmethod
    def _is_legal_article_start(text: str) -> bool:
        patterns = [
            r'^第[零一二三四五六七八九十百千\d]+条',
            r'^第[零一二三四五六七八九十百千\d]+款',
            r'^第[零一二三四五六七八九十百千\d]+章',
            r'^第[零一二三四五六七八九十百千\d]+节',
        ]
        return any(re.match(pattern, text) for pattern in patterns)
    
    @staticmethod
    def parse_file(file_path: str) -> List[DocumentChunk]:
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return DocumentParser.parse_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return DocumentParser.parse_word(file_path)
        elif ext in ['.md', '.markdown']:
            return DocumentParser.parse_markdown(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
