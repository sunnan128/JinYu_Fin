# ── 金语AI 金融知识问答系统 前端 ──
# 设计决策：
# - 品牌名 金语AI（JinYu）：金语 = 金融智慧之语，与 LexAI 同系列产品线
# - 配色：深绿色(#14513b) + 主绿色(#1f8a5f) + 薄荷绿(#34a06b)，洁净米白底(#f4f8f6)
# - 衬线字体 Noto Serif SC（标题）保持专业感，Inter（正文）保持清晰
# - 卡片式布局，引用来源 hover 高亮绿色边框
# - 新增「非金融建议」常驻免责声明（金融场景合规必需）

import streamlit as st
import requests
import json
import time
import re
import subprocess
import sys
import os
from datetime import datetime
from guard_banner import build_guard_banner_html

API_URL = "http://localhost:8006"

# ── Page Config ──
st.set_page_config(
    page_title="金语AI · 金融知识问答系统",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS: 洁净金语风 ──
st.markdown("""
<style>
    /* ── 全局基调 ── */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    * { font-family: 'Inter', 'Noto Serif SC', -apple-system, sans-serif; }

    .stApp {
        background: #f4f8f6;
    }

    /* ── 主标 ── */
    .hero-title {
        font-family: 'Noto Serif SC', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #14513b;
        letter-spacing: 0.02em;
        margin-bottom: 0.25rem;
        line-height: 1.3;
    }
    .hero-sub {
        color: #6b8a7e;
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 0.04em;
        margin-bottom: 1.2rem;
        border-left: 3px solid #1f8a5f;
        padding-left: 1rem;
    }

    /* ── 免责声明横幅 ── */
    .disclaimer {
        background: #eafaf1;
        border: 1px solid #b7e6cd;
        border-left: 4px solid #1f8a5f;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #1c352c;
        line-height: 1.6;
        margin-bottom: 1.4rem;
    }
    .disclaimer strong { color: #14513b; }

    /* ── L3 幻觉抑制：⚠️ 时效风险提示条 ── */
    .guard-banner {
        background: #fff8ec;
        border: 1px solid #f0c674;
        border-left: 4px solid #e8a33d;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.85rem;
        color: #6b4e1f;
        line-height: 1.6;
    }
    .guard-banner .guard-title {
        font-family: 'Noto Serif SC', serif;
        font-weight: 700;
        font-size: 0.95rem;
        color: #b9740f;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .guard-banner .guard-body { margin-bottom: 0.4rem; }
    .guard-banner .guard-em { color: #b9740f; font-weight: 600; }
    .guard-banner .guard-list {
        margin: 0.3rem 0 0 0;
        padding-left: 1.2rem;
    }
    .guard-banner .guard-list li {
        margin-bottom: 0.3rem;
        color: #6b4e1f;
    }
    .guard-banner .guard-list strong { color: #8a5a11; }
    .guard-banner .guard-details {
        margin-top: 0.25rem;
        display: flex;
        flex-direction: column;
        gap: 0.1rem;
    }
    .guard-banner .guard-detail {
        font-size: 0.8rem;
        color: #8a6a2f;
    }
    .guard-banner .guard-detail a {
        color: #1a5fb4;
        font-weight: 600;
        text-decoration: underline;
    }
    .guard-banner .guard-detail a:hover { color: #0b3d91; }

    /* ── 卡片标题（通用） ── */
    .card-title {
        font-family: 'Noto Serif SC', serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #14513b;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #e3efe9;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── 可折叠抽屉样式 ── */
    div[data-testid="stExpander"] {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    div[data-testid="stExpander"] summary {
        font-family: 'Noto Serif SC', serif;
        font-weight: 600;
        color: #14513b;
        padding: 0.3rem 0;
    }

    /* ── 问答区域 ── */
    .answer-box {
        background: #fbfdfc;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #1f8a5f;
        margin: 1rem 0;
        line-height: 1.8;
        font-size: 0.95rem;
        color: #1c352c;
    }
    .answer-box strong {
        color: #14513b;
    }
    .answer-box.safe {
        background: #fff5f5;
        border-left-color: #e25555;
    }

    /* ── 引用来源样式 ── */
    .citation-item {
        background: #ffffff;
        border: 1px solid #e3efe9;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s;
    }
    .citation-item:hover {
        border-color: #1f8a5f;
    }
    .citation-item a:hover {
        color: #14513b !important;
        text-decoration: underline !important;
    }
    .citation-meta {
        font-size: 0.8rem;
        color: #9bb3a8;
        display: flex;
        gap: 1.5rem;
        margin-top: 0.4rem;
    }
    .citation-meta span {
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* ── 侧栏 ── */
    .sidebar-content {
        padding: 0.5rem 0;
    }
    .sidebar-section {
        font-size: 0.75rem;
        font-weight: 600;
        color: #9bb3a8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.2rem 0 0.6rem 0;
    }

    /* ── 状态指示器 ── */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-dot.online { background: #10b981; }
    .status-dot.offline { background: #ef4444; }

    /* ── 文件上传区 ── */
    .uploaded-file-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid #eef4f1;
    }
    .uploaded-file-row:last-child { border-bottom: none; }

    /* ── 覆盖 Streamlit 默认样式 ── */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"] {
        background: #1f8a5f !important;
        border: none !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #176b49 !important;
        box-shadow: 0 2px 8px rgba(31,138,95,0.25);
    }
    /* ── 上传中禁用态 ── */
    .stButton > button[kind="primary"]:disabled,
    .stButton > button:disabled {
        opacity: 0.45 !important;
        cursor: not-allowed !important;
        background: #9bb3a8 !important;
        box-shadow: none !important;
    }
    .stButton > button[kind="primary"]:disabled:hover {
        background: #9bb3a8 !important;
        box-shadow: none !important;
    }
    div[data-testid="stTextInput"] input {
        border-radius: 8px;
        border: 1px solid #e3efe9;
        padding: 0.6rem 1rem;
        font-size: 0.9rem;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #1f8a5f;
        box-shadow: 0 0 0 2px rgba(31,138,95,0.15);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #eafaf1;
        border-radius: 10px;
        padding: 3px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        font-size: 0.85rem;
        color: #6b8a7e;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff;
        color: #14513b;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stSpinner > div {
        border-top-color: #1f8a5f !important;
    }

    /* ── 处理时间标签 ── */
    .meta-tag {
        display: inline-flex;
        align-items: center;
        background: #eafaf1;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        color: #6b8a7e;
        gap: 0.3rem;
    }

    /* ── 空状态 ── */
    .empty-state {
        text-align: center;
        padding: 2.5rem 1rem;
        color: #9bb3a8;
        font-size: 0.9rem;
    }
    .empty-state-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
        opacity: 0.5;
    }

    /* ── 工具栏中文化 ── */
    .stStatusWidget button[data-testid="stBaseButton-header"] span {
        font-size: 0;
        position: relative;
        display: inline-block;
    }
    .stStatusWidget button[data-testid="stBaseButton-header"] span::after {
        font-size: 0.8rem;
        content: "停止";
        position: absolute;
        left: 0;
        top: 0;
        color: inherit;
        white-space: nowrap;
    }

    /* ── footer ── */
    .footer-note {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        font-size: 0.7rem;
        color: #b6ccc2;
        letter-spacing: 0.05em;
    }

    /* ══════════════════════════════════════════
       文档管理右列独立滚动（JS 注入主导，CSS 兜底）
       ══════════════════════════════════════════ */

    .doc-card-header {
        font-family: 'Noto Serif SC', serif;
        font-size: 1rem;
        font-weight: 600;
        color: #14513b;
        padding-bottom: 0.5rem;
        margin-bottom: 0.25rem;
        border-bottom: 2px solid #e3efe9;
    }

    /* ── 滚动列容器样式（JS 会将该类加到右列 column 上） ── */
    .scroll-col {
        background: #ffffff !important;
        border: 1px solid #e3efe9 !important;
        border-radius: 10px !important;
        padding: 0.75rem 0.75rem 0.25rem !important;
        max-height: 70vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin;       /* Firefox */
        scrollbar-color: #d0dfd7 transparent;
    }
    .scroll-col::-webkit-scrollbar {
        width: 6px;
    }
    .scroll-col::-webkit-scrollbar-track {
        background: transparent;
    }
    .scroll-col::-webkit-scrollbar-thumb {
        background: #d0dfd7;
        border-radius: 3px;
        opacity: 0.6;
        transition: opacity 0.2s ease;
    }
    .scroll-col::-webkit-scrollbar-thumb:hover {
        background: #9bb3a8;
        opacity: 1;
    }

    /* ── 文档条目行 ── */
    .doc-row-name {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1c352c;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .doc-row-meta {
        font-size: 0.72rem;
        color: #9bb3a8;
        margin-top: 1px;
    }

    /* ── 卡片空状态 ── */
    .doc-card-empty {
        text-align: center;
        padding: 2rem 0.5rem;
        color: #9bb3a8;
        font-size: 0.85rem;
    }
    .doc-card-empty-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        opacity: 0.5;
    }

    /* ── 响应式：移动端 ── */
    @media (max-width: 768px) {
        .scroll-col {
            max-height: 35vh !important;
        }
    }

    /* ── 文件上传区加大 ── */
    section[data-testid="stFileUploaderDropzone"] {
        min-height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──

def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False

def upload_document_async(file, progress_bar, status_text):
    """异步上传文件并轮询实时进度，返回上传结果"""
    try:
        files = {"file": (file.name, file, file.type)}
        r = requests.post(f"{API_URL}/upload/start", files=files)
        if r.status_code != 200:
            st.error(f"上传失败：{r.json().get('detail', '未知错误')}")
            return None
        task_id = r.json()["task_id"]

        while True:
            time.sleep(0.5)
            retries = 0
            while retries < 3:
                pr = requests.get(f"{API_URL}/upload/progress/{task_id}")
                if pr.status_code == 200:
                    break
                retries += 1
                time.sleep(1.0)
            
            if pr.status_code != 200:
                # 重试耗尽：后端可能热重启导致进度丢失，尝试直接查文档列表
                st.warning("进度数据暂不可用，正在检查文档是否已上传完成…")
                time.sleep(2)
                docs = list_documents()
                for d in docs:
                    if d.get("filename") == file.name:
                        st.info(f"✅ {file.name} 已上传完成（共 {d['chunk_count']} 个片段）")
                        return {"filename": file.name, "chunk_count": d["chunk_count"]}
                st.error("进度查询失败，请稍后查看文档列表确认上传状态。")
                return None

            data = pr.json()
            progress = data["progress"]
            stage = data["stage"]
            message = data["message"]

            if progress >= 0:
                progress_bar.progress(min(progress, 1.0))
            status_text.text(f"📄 {file.name} — {message}")

            if stage == "done":
                match = re.search(r'(\d+) 个片段', message)
                chunk_count = int(match.group(1)) if match else 0
                return {"filename": file.name, "chunk_count": chunk_count}
            elif stage == "error":
                st.error(f"{file.name} 处理失败：{message}")
                return None
            elif stage == "recovered":
                # 后端热重启导致进度丢失，检查文档是否已上传完成
                time.sleep(2)
                docs = list_documents()
                for d in docs:
                    if d.get("filename") == file.name:
                        st.info(f"✅ {file.name} 已上传完成（共 {d['chunk_count']} 个片段）")
                        return {"filename": file.name, "chunk_count": d["chunk_count"]}
                st.warning("进度数据暂不可用，请刷新页面后查看文档列表。")
                return None
    except Exception as e:
        st.error(f"上传失败：{str(e)}")
        return None

def list_documents():
    try:
        r = requests.get(f"{API_URL}/documents")
        return r.json() if r.status_code == 200 else []
    except:
        return []

def delete_document(document_id):
    try:
        r = requests.delete(f"{API_URL}/documents/{document_id}")
        return r.status_code == 200
    except:
        return False

def query_question(question, top_k=5, use_rerank=True, use_keyword=True):
    try:
        payload = {
            "question": question,
            "top_k": top_k,
            "use_rerank": use_rerank,
            "use_keyword_search": use_keyword
        }
        r = requests.post(f"{API_URL}/query", json=payload)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"查询失败：{r.json().get('detail', '未知错误')}")
            return None
    except Exception as e:
        st.error(f"查询失败：{str(e)}")
        return None


# ── 检查后端状态 ──
health_ok = check_health()


# ═══════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

    st.markdown("### 💰 金语AI")
    st.caption("Financial Intelligence System")

    st.markdown('<div class="sidebar-section">系统状态</div>', unsafe_allow_html=True)
    if health_ok:
        st.markdown('<span class="status-dot online"></span> 服务运行中', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot offline"></span> 服务未连接', unsafe_allow_html=True)

    col1, col2 = st.columns([0.65, 0.35])
    with col2:
        if st.button("🔄 重连", use_container_width=True, help="重启后端服务"):
            with st.spinner("正在重启后端..."):
                restart_script = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "restart_backend.py"
                )
                result = subprocess.run(
                    [sys.executable, restart_script],
                    capture_output=True, text=True, timeout=60
                )
            if result.returncode == 0:
                st.success("✅ 后端已自动恢复")
                time.sleep(2)
                st.rerun()
            else:
                log = (result.stdout or "") + (result.stderr or "")
                if log.strip():
                    st.error("❌ 自动恢复失败，详情如下：")
                    # 用大文本域替代 st.code，支持滚动查看完整日志
                    st.text_area(
                        label="完整日志输出",
                        value=log.strip(),
                        height=400,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    # 底部显示一些统计信息
                    lines = log.strip().split('\n')
                    err_lines = [l for l in lines if 'ERR' in l or '失败' in l or '❌' in l]
                    if err_lines:
                        st.caption(f"共 {len(lines)} 行，其中 {len(err_lines)} 个错误/警告")
                else:
                    st.error("❌ 自动恢复失败（无详细日志输出）")

    st.markdown('<div class="sidebar-section">关于系统</div>', unsafe_allow_html=True)
    st.markdown("""
    基于 **RAG + 混合检索 + 知识图谱** 构建的专业金融知识引擎。

    - 语义检索 · 关键词检索 · 图谱检索
    - 溯源至报告章节 / 附件页码
    - 未检索到即明确告知
    - 杜绝生成式幻觉
    """)

    st.markdown('<div class="sidebar-section">技术支持</div>', unsafe_allow_html=True)
    st.caption("FastAPI · ChromaDB · Neo4j · SentenceTransformer · DeepSeek")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# 主页面
# ═══════════════════════════════════════════════

col_logo, col_title = st.columns([0.06, 1])
with col_logo:
    st.markdown("<div style='font-size:2.6rem;margin-top:0.2rem;'>💰</div>", unsafe_allow_html=True)
with col_title:
    st.markdown('<div class="hero-title">金语AI 金融知识问答系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">RAG 增强 · 知识图谱 · 溯源可信 · 安全合规</div>', unsafe_allow_html=True)

# 常驻免责声明（金融场景合规必需）
st.markdown(
    '<div class="disclaimer">'
    '⚠️ <strong>免责声明：</strong>本系统仅基于公开金融资料提供检索与参考，'
    '不构成投资建议。具体决策请咨询专业金融顾问。'
    '</div>',
    unsafe_allow_html=True
)

if not health_ok:
    st.warning("⚠ 后端服务尚未连接，请确认服务已启动。")

tab_doc, tab_qa = st.tabs(["📂 文档管理", "💡 金融检索"])

# ── 上传模块 session state ──
# upload_key_counter: 每完成一批上传后递增，用于重置 file_uploader widget（释放文件名额）
# processed_files: 最近成功解析的文件列表，供可视化回显
# uploading: 上传锁——True 表示有任务正在执行，禁用按钮防止重复点击
if "upload_key_counter" not in st.session_state:
    st.session_state.upload_key_counter = 0
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "uploading" not in st.session_state:
    st.session_state.uploading = False
if "confirm_del" not in st.session_state:
    st.session_state.confirm_del = None

BATCH_LIMIT = 5  # 单批次最多处理的文件数


# ═══════════════════════════════════════════════
# Tab 1：文档管理
# ═══════════════════════════════════════════════

with tab_doc:
    col_left, col_right = st.columns([5, 7])

    with col_left:
        st.markdown('<div class="card-title" style="border-bottom:none;margin-bottom:0.5rem;">📄 上传金融资料</div>', unsafe_allow_html=True)

        # 动态 key 确保每批上传完成后 widget 自动重置（旧文件不再保留）
        uploaded_files = st.file_uploader(
            "选择金融文档（PDF / Word / Markdown，单批最多 5 份，可分批连续上传）",
            type=["pdf", "docx", "doc", "md", "markdown"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"file_uploader_{st.session_state.upload_key_counter}"
        )

        if uploaded_files:
            # 不超过限制：正常显示与上传
            if len(uploaded_files) <= BATCH_LIMIT:
                # ── 上传中锁定提示 ──
                if st.session_state.uploading:
                    st.warning("⏳ 文件正在上传解析中，请等待完成后再操作", icon="🔄")
                st.caption(f"已选 {len(uploaded_files)} 份文件")

                if st.button(
                    "上传并解析",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.uploading
                ):
                    st.session_state.uploading = True
                    overall_bar = st.progress(0, text="准备中…")
                    detail_status = st.empty()
                    results = []

                    for i, f in enumerate(uploaded_files):
                        overall_bar.progress(
                            i / len(uploaded_files),
                            text=f"文件 ({i+1}/{len(uploaded_files)})"
                        )
                        file_bar = st.progress(0.0, text=f"⏳ {f.name}")

                        result = upload_document_async(f, file_bar, detail_status)
                        if result:
                            results.append(result)
                        file_bar.empty()

                    overall_bar.empty()
                    detail_status.empty()

                    if results:
                        total_chunks = sum(r['chunk_count'] for r in results)
                        now_str = datetime.now().strftime("%H:%M:%S")
                        for r in results:
                            st.session_state.processed_files.append({
                                "filename": r["filename"],
                                "chunk_count": r["chunk_count"],
                                "time": now_str,
                            })
                        st.success(f"✅ {len(results)} 份文件全部解析完成，共 {total_chunks} 个片段")
                        st.session_state.upload_key_counter += 1
                    else:
                        st.warning("⚠ 文件均上传失败，请检查后端服务后重试")
                    # 无论成功或失败，释放上传锁
                    st.session_state.uploading = False
                    time.sleep(1.2)
                    st.rerun()
            else:
                # 用户一次选了超过 BATCH_LIMIT 份：不硬拦截，自动分批处理
                batch_files = uploaded_files[:BATCH_LIMIT]
                remaining = len(uploaded_files) - BATCH_LIMIT
                # ── 上传中锁定提示 ──
                if st.session_state.uploading:
                    st.warning("⏳ 文件正在上传解析中，请等待完成后再操作", icon="🔄")
                st.info(
                    f"本次处理前 {BATCH_LIMIT} 份，剩余 {remaining} 份请点击「上传并解析」后再次选择即可自动续传",
                    icon="🔄"
                )
                st.caption(f"当前批次处理 {BATCH_LIMIT} 份，待处理 {remaining} 份")

                if st.button(
                    "上传并解析（前5份）",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.uploading
                ):
                    st.session_state.uploading = True
                    overall_bar = st.progress(0, text="准备中…")
                    detail_status = st.empty()
                    results = []

                    for i, f in enumerate(batch_files):
                        overall_bar.progress(
                            i / len(batch_files),
                            text=f"文件 ({i+1}/{len(batch_files)})"
                        )
                        file_bar = st.progress(0.0, text=f"⏳ {f.name}")

                        result = upload_document_async(f, file_bar, detail_status)
                        if result:
                            results.append(result)
                        file_bar.empty()

                    overall_bar.empty()
                    detail_status.empty()

                    if results:
                        total_chunks = sum(r['chunk_count'] for r in results)
                        now_str = datetime.now().strftime("%H:%M:%S")
                        for r in results:
                            st.session_state.processed_files.append({
                                "filename": r["filename"],
                                "chunk_count": r["chunk_count"],
                                "time": now_str,
                            })
                        st.success(f"✅ 前 {len(results)} 份解析完成（{total_chunks} 个片段），剩余 {remaining} 份请重新选择上传")
                        st.session_state.upload_key_counter += 1
                    else:
                        st.warning("⚠ 文件均上传失败，请检查后端服务后重试")
                    st.session_state.uploading = False
                    time.sleep(1.2)
                    st.rerun()
        else:
            # 没有选中文件时：显示提示 + 最近解析记录
            st.info("支持 PDF、Word 和 Markdown 格式金融资料（研报 / 公告），单批最多 5 份，可分批连续上传。", icon="ℹ️")

            # ── 最近解析完成可视化 ──
            if st.session_state.processed_files:
                st.markdown(
                    '<div style="margin-top:1rem;padding:0.75rem;background:#eafaf1;'
                    'border-radius:8px;border-left:3px solid #1f8a5f;">'
                    '<div style="font-size:0.8rem;font-weight:600;color:#14513b;margin-bottom:0.4rem;">'
                    '📌 最近解析完成</div>',
                    unsafe_allow_html=True
                )
                # 显示最近 6 条
                for pf in st.session_state.processed_files[-6:]:
                    st.markdown(
                        f'<div style="font-size:0.8rem;color:#1c352c;padding:0.15rem 0;">'
                        f'<span style="color:#1f8a5f;">✓</span> {pf["filename"]} '
                        f'<span style="color:#9bb3a8;font-size:0.7rem;">({pf["chunk_count"]} 片段 · {pf["time"]})</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                if len(st.session_state.processed_files) > 6:
                    st.markdown(
                        f'<div style="font-size:0.7rem;color:#9bb3a8;margin-top:0.2rem;">'
                        f'… 共 {len(st.session_state.processed_files)} 份已解析</div>',
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="doc-card-header">📚 已入库文档</div>', unsafe_allow_html=True)

        # ── 搜索输入框 ──
        search_text = st.text_input(
            "搜索文件名",
            placeholder="🔍 输入文件名模糊匹配…",
            label_visibility="collapsed",
            key="doc_search"
        )

        # ── 独立滚动容器（Streamlit 原生） ──
        with st.container(height=550, border=False):
            # ── 获取并筛选文档列表 ──
            documents = list_documents()
            if search_text:
                search_lower = search_text.lower()
                documents = [d for d in documents if search_lower in d['filename'].lower()]

            if documents:
                for doc in documents:
                    upload_time = datetime.fromisoformat(doc['upload_time'].replace('Z', '+00:00'))
                    doc_id = doc['id']

                    # ── 两段确认删除 ──
                    if st.session_state.confirm_del == doc_id:
                        col_name, col_yes, col_no = st.columns([4, 1, 1])
                        with col_name:
                            st.markdown(f"""
                            <div>
                                <div class="doc-row-name" title="{doc['filename']}">{doc['filename']}</div>
                                <div class="doc-row-meta">{doc['chunk_count']} 个片段 · {upload_time.strftime('%Y-%m-%d %H:%M')}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_yes:
                            if st.button("✓ 确认", key=f"yes_{doc_id}", type="primary", use_container_width=True):
                                if delete_document(doc_id):
                                    st.toast(f"已删除「{doc['filename']}」", icon="🗑️")
                                    st.session_state.confirm_del = None
                                    st.rerun()
                        with col_no:
                            if st.button("✗ 取消", key=f"no_{doc_id}", use_container_width=True):
                                st.session_state.confirm_del = None
                                st.rerun()
                    else:
                        col_name, col_del = st.columns([5, 1])
                        with col_name:
                            st.markdown(f"""
                            <div>
                                <div class="doc-row-name" title="{doc['filename']}">{doc['filename']}</div>
                                <div class="doc-row-meta">{doc['chunk_count']} 个片段 · {upload_time.strftime('%Y-%m-%d %H:%M')}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_del:
                            if st.button("删除", key=f"del_{doc_id}", use_container_width=True):
                                st.session_state.confirm_del = doc_id
                                st.rerun()
            else:
                st.markdown(f"""
                <div class="doc-card-empty">
                    <div class="doc-card-empty-icon">📋</div>
                    {"暂无匹配文件" if search_text else "暂无入库文档"}
                    <br><span style="font-size:0.8rem;">请先在左侧上传金融资料</span>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# Tab 2：金融检索
# ═══════════════════════════════════════════════

if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "search_clicked" not in st.session_state:
    st.session_state.search_clicked = False
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

with tab_qa:
    st.markdown('<div class="card-title" style="border-bottom:none;margin-bottom:0.5rem;">🔍 金融知识检索</div>', unsafe_allow_html=True)

    with st.expander("检索参数设置", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            top_k = st.slider("召回数量", 1, 10, 5, help="检索返回的关联文档片段数")
        with c2:
            use_rerank = st.checkbox("关联重排序", value=True, help="对检索结果按相关性二次排序")
        with c3:
            use_keyword = st.checkbox("关键词模式", value=True, help="同时启用关键词精确匹配")

    question = st.text_input(
        "## 输入金融问题",
        placeholder="例：科创板上市条件有哪些？",
        label_visibility="collapsed",
        key=f"qa_input_{st.session_state.input_key}"
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        search_btn = st.button("🔍 检索", type="primary", use_container_width=True)
    with btn_col2:
        clear_btn = st.button("清空", use_container_width=True)

    if search_btn:
        st.session_state.search_clicked = True

    if clear_btn:
        st.session_state.last_question = ""
        st.session_state.last_result = None
        st.session_state.search_clicked = False
        st.session_state.input_key += 1
        st.rerun()

    trigger_search = (question and question != st.session_state.last_question) or \
                     (question and st.session_state.search_clicked)

    if trigger_search:
        st.session_state.last_question = question
        st.session_state.search_clicked = False
        with st.spinner("正在检索金融知识库，请稍候…"):
            result = query_question(question, top_k, use_rerank, use_keyword)
        if result:
            st.session_state.last_result = {"question": question, "result": result}

    if st.session_state.last_result:
        qa = st.session_state.last_result
        st.markdown(f'<div style="font-family:\'Noto Serif SC\',serif;font-size:0.95rem;font-weight:600;color:#14513b;background:#eafaf1;border-radius:8px;padding:0.5rem 1rem;margin:0.5rem 0 0.25rem 0;">📝 {qa["question"]}</div>', unsafe_allow_html=True)

        # ── L3 幻觉抑制：时效风险提示条（仅当后端 guard.blocked 为真时展示） ──
        guard_html = build_guard_banner_html(qa['result'].get('guard'))
        if guard_html:
            st.markdown(guard_html, unsafe_allow_html=True)

        found = qa['result']['found_in_knowledge_base']
        answer = qa['result']['answer']
        # 命中：普通绿框；未命中 / 安全拒答：红色安全框
        if found:
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="answer-box safe">⚠ {answer}<br><br>本内容基于公开金融资料生成，仅供学习参考，不构成投资建议。</div>', unsafe_allow_html=True)

        if qa['result']['citations']:
            st.markdown("#### 📎 引用溯源")
            for cite in qa['result']['citations']:
                location = []
                if cite['page_number']:
                    location.append(f"第 {cite['page_number']} 页")
                if cite['paragraph_number']:
                    location.append(f"第 {cite['paragraph_number']} 段")
                loc_str = " · ".join(location) if location else "全文检索"
                doc_id = cite.get('document_id', '')
                cite_para = cite.get('paragraph_number', '')
                view_url = f"{API_URL}/documents/{doc_id}/view"
                if cite_para:
                    view_url += f"?page=1&para={cite_para}"

                st.markdown(f"""
                <div class="citation-item">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <strong style="font-size:0.9rem;">{cite['document_name']}</strong>
                        <a href="{view_url}" target="_blank"
                           style="font-size:0.75rem;color:#1f8a5f;text-decoration:none;white-space:nowrap;margin-left:8px;"
                           title="查看该文档所有检索片段">
                           📄 查看原文 →
                        </a>
                    </div>
                    <div class="citation-meta">
                        <span>📖 {loc_str}</span>
                        <span>🎯 相关度 {cite['score']:.4f}</span>
                    </div>
                    <div style="font-size:0.85rem;color:#3f5a4f;margin-top:0.4rem;line-height:1.6;">
                        {cite['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f'<span class="meta-tag">⏱ {qa["result"]["processing_time_ms"]:.0f} ms</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer-note">金语AI Financial Intelligence · 严谨 · 专业 · 合规 · 仅供学习参考</div>', unsafe_allow_html=True)
