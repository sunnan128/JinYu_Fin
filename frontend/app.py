# ── LexAI 法律智能检索系统 前端 ──
# 设计决策：
# - 品牌名 LexAI：Lex（拉丁语"法律"）+ AI
# - 配色：深蓝灰(#1a1a2e) + 金色(#c9a84c)，米白底(#f8f7f4)
# - 衬线字体 Noto Serif SC，契合法律文书传统感
# - 卡片式布局，引用来源 hover 高亮金色边框

import streamlit as st
import requests
import json
import time
import re
import subprocess
import sys
import os
from datetime import datetime

API_URL = "http://localhost:8006"

# ── Page Config ──
st.set_page_config(
    page_title="LexAI · 法律智能检索系统",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS: 高智法律风格 ──
st.markdown("""
<style>
    /* ── 全局基调 ── */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    * { font-family: 'Inter', 'Noto Serif SC', -apple-system, sans-serif; }

    .stApp {
        background: #f8f7f4;
    }

    /* ── 主标 ── */
    .hero-title {
        font-family: 'Noto Serif SC', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        letter-spacing: 0.02em;
        margin-bottom: 0.25rem;
        line-height: 1.3;
    }
    .hero-sub {
        color: #6b7280;
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 0.04em;
        margin-bottom: 2rem;
        border-left: 3px solid #c9a84c;
        padding-left: 1rem;
    }

    /* ── 卡片标题（通用） ── */
    .card-title {
        font-family: 'Noto Serif SC', serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #f0efe9;
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
        color: #1a1a2e;
        padding: 0.3rem 0;
    }

    /* ── 问答区域 ── */
    .answer-box {
        background: #fafaf7;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #c9a84c;
        margin: 1rem 0;
        line-height: 1.8;
        font-size: 0.95rem;
        color: #1f2937;
    }
    .answer-box strong {
        color: #1a1a2e;
    }

    /* ── 引用来源样式 ── */
    .citation-item {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s;
    }
    .citation-item:hover {
        border-color: #c9a84c;
    }
    .citation-item a:hover {
        color: #1a1a2e !important;
        text-decoration: underline !important;
    }
    .citation-meta {
        font-size: 0.8rem;
        color: #9ca3af;
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
        color: #9ca3af;
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
        border-bottom: 1px solid #f3f4f6;
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
        background: #1a1a2e !important;
        border: none !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2d2d4a !important;
        box-shadow: 0 2px 8px rgba(26,26,46,0.2);
    }
    div[data-testid="stTextInput"] input {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        padding: 0.6rem 1rem;
        font-size: 0.9rem;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #c9a84c;
        box-shadow: 0 0 0 2px rgba(201,168,76,0.15);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #f3f2ee;
        border-radius: 10px;
        padding: 3px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        font-size: 0.85rem;
        color: #6b7280;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff;
        color: #1a1a2e;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stSpinner > div {
        border-top-color: #c9a84c !important;
    }

    /* ── 处理时间标签 ── */
    .meta-tag {
        display: inline-flex;
        align-items: center;
        background: #f3f2ee;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        color: #6b7280;
        gap: 0.3rem;
    }

    /* ── 空状态 ── */
    .empty-state {
        text-align: center;
        padding: 2.5rem 1rem;
        color: #9ca3af;
        font-size: 0.9rem;
    }
    .empty-state-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
        opacity: 0.5;
    }

    /* ── 工具栏中文化 ── */
    /* "Stop" → "停止" */
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
        color: #d1d5db;
        letter-spacing: 0.05em;
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
        # Step 1: 启动异步上传，获取 task_id
        files = {"file": (file.name, file, file.type)}
        r = requests.post(f"{API_URL}/upload/start", files=files)
        if r.status_code != 200:
            st.error(f"上传失败：{r.json().get('detail', '未知错误')}")
            return None
        task_id = r.json()["task_id"]
        
        # Step 2: 轮询进度
        while True:
            time.sleep(0.5)
            pr = requests.get(f"{API_URL}/upload/progress/{task_id}")
            if pr.status_code != 200:
                st.error("进度查询失败")
                return None
            
            data = pr.json()
            progress = data["progress"]
            stage = data["stage"]
            message = data["message"]
            
            # 更新进度条和状态文字
            if progress >= 0:
                progress_bar.progress(min(progress, 1.0))
            status_text.text(f"📄 {file.name} — {message}")
            
            # 完成或出错
            if stage == "done":
                # 从 message 中提取 chunk_count
                match = re.search(r'(\d+) 个片段', message)
                chunk_count = int(match.group(1)) if match else 0
                return {"filename": file.name, "chunk_count": chunk_count}
            elif stage == "error":
                st.error(f"{file.name} 处理失败：{message}")
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

    st.markdown("### ⚖ LexAI")
    st.caption("Legal Intelligence System")

    st.markdown('<div class="sidebar-section">系统状态</div>', unsafe_allow_html=True)
    if health_ok:
        st.markdown('<span class="status-dot online"></span> 服务运行中', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot offline"></span> 服务未连接', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">关于系统</div>', unsafe_allow_html=True)
    st.markdown("""
    基于 **RAG + 混合检索** 构建的专业法律知识引擎。

    - 语义检索 · 关键词检索
    - 溯源至原文页码段落
    - 未检索到即明确告知
    - 杜绝生成式幻觉
    """)

    st.markdown('<div class="sidebar-section">技术支持</div>', unsafe_allow_html=True)
    st.caption("FastAPI · ChromaDB · SentenceTransformer · DeepSeek")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# 主页面
# ═══════════════════════════════════════════════

col_logo, col_title = st.columns([0.06, 1])
with col_logo:
    st.markdown("<div style='font-size:2.6rem;margin-top:0.2rem;'>⚖</div>", unsafe_allow_html=True)
with col_title:
    st.markdown('<div class="hero-title">LexAI 法律智能检索系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">RAG 增强 · 混合检索 · 溯源可信 · 零幻觉</div>', unsafe_allow_html=True)

if not health_ok:
    col_warn, col_btn = st.columns([0.7, 0.3])
    with col_warn:
        st.warning("⚠ 后端服务尚未连接，请确认服务已启动。")
    with col_btn:
        if st.button("🔄 尝试恢复连接", use_container_width=True):
            with st.spinner("正在尝试重启后端服务..."):
                restart_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "restart_backend.py")
                result = subprocess.run(
                    [sys.executable, restart_script],
                    capture_output=True, text=True, timeout=60
                )
            if result.returncode == 0:
                st.success("✅ 后端服务已自动恢复，正在刷新...")
                time.sleep(2)
                st.rerun()
            else:
                st.error(f"❌ 自动恢复失败，请手动启动后端:\n{result.stderr}")

tab_doc, tab_qa = st.tabs(["📂 文档管理", "💡 法律检索"])


# ═══════════════════════════════════════════════
# Tab 1：文档管理
# ═══════════════════════════════════════════════

with tab_doc:
    col_left, col_right = st.columns([5, 7])

    with col_left:
        st.markdown('<div class="card-title" style="border-bottom:none;margin-bottom:0.5rem;">📄 上传卷宗</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "选择法律文档（PDF / Word，最多 5 份）",
            type=["pdf", "docx", "doc"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            if len(uploaded_files) > 5:
                st.error(f"最多上传 5 份文件，当前选了 {len(uploaded_files)} 份")
            else:
                st.caption(f"已选 {len(uploaded_files)} 份文件")
            if st.button("上传并解析", type="primary", use_container_width=True):
                overall_bar = st.progress(0, text="准备中…")
                detail_status = st.empty()
                results = []
                
                for i, f in enumerate(uploaded_files):
                    # 更新整体进度
                    overall_bar.progress(
                        i / len(uploaded_files),
                        text=f"文件 ({i+1}/{len(uploaded_files)})"
                    )
                    # 当前文件的实时进度条（嵌套）
                    file_bar = st.progress(0.0, text=f"⏳ {f.name}")
                    
                    result = upload_document_async(f, file_bar, detail_status)
                    if result:
                        results.append(result)
                    file_bar.empty()
                
                overall_bar.empty()
                detail_status.empty()
                
                if results:
                    total_chunks = sum(r['chunk_count'] for r in results)
                    st.success(f"✅ {len(results)} 份文件全部解析完成，共 {total_chunks} 个片段")
                    st.rerun()
        else:
            st.info("支持 PDF、DOCX 格式法律文书，最多 5 份同时上传。", icon="ℹ️")

    with col_right:
        with st.expander("📚 已入库文档", expanded=True):
            documents = list_documents()

            if documents:
                for doc in documents:
                    upload_time = datetime.fromisoformat(doc['upload_time'].replace('Z', '+00:00'))
                    st.markdown(f"""
                    <div class="uploaded-file-row">
                        <div>
                            <strong style="font-size:0.9rem;">{doc['filename']}</strong>
                            <div style="font-size:0.75rem;color:#9ca3af;margin-top:2px;">
                                {doc['chunk_count']} 个片段 · {upload_time.strftime('%Y-%m-%d %H:%M')}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("删除", key=f"del_{doc['id']}", type="secondary"):
                        if delete_document(doc['id']):
                            st.success("已移除")
                            st.rerun()
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">📋</div>
                    暂无入库文档<br>
                    <span style="font-size:0.8rem;">请先在左侧上传法律卷宗</span>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# Tab 2：法律检索
# ═══════════════════════════════════════════════

# 初始化上次提交记录和最新结果
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "search_clicked" not in st.session_state:
    st.session_state.search_clicked = False
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

with tab_qa:
    st.markdown('<div class="card-title" style="border-bottom:none;margin-bottom:0.5rem;">🔍 法律条文检索</div>', unsafe_allow_html=True)

    with st.expander("检索参数设置", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            top_k = st.slider("召回数量", 1, 10, 5, help="检索返回的关联文档片段数")
        with c2:
            use_rerank = st.checkbox("关联重排序", value=True, help="对检索结果按相关性二次排序")
        with c3:
            use_keyword = st.checkbox("关键词模式", value=True, help="同时启用关键词精确匹配")

    # ── 输入框（按回车 或 点击"检索"按钮均可触发） ──
    question = st.text_input(
        "## 输入法律问题",
        placeholder="例：民法典第584条关于违约损害赔偿的范围如何规定？",
        label_visibility="collapsed",
        key=f"qa_input_{st.session_state.input_key}"
    )

    # 检索按钮（位于输入框下方，均匀分布）
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

    # 检测新提交：按回车（内容变化）或 点击"检索"按钮
    trigger_search = (question and question != st.session_state.last_question) or \
                     (question and st.session_state.search_clicked)
    
    if trigger_search:
        st.session_state.last_question = question
        st.session_state.search_clicked = False
        with st.spinner("正在检索知识库，请稍候…"):
            result = query_question(question, top_k, use_rerank, use_keyword)
        if result:
            st.session_state.last_result = {"question": question, "result": result}

    # ── 显示当前检索结果（仅最新一次） ──
    if st.session_state.last_result:
        qa = st.session_state.last_result
        st.markdown(f'<div style="font-family:\'Noto Serif SC\',serif;font-size:0.95rem;font-weight:600;color:#1a1a2e;background:#f3f2ee;border-radius:8px;padding:0.5rem 1rem;margin:0.5rem 0 0.25rem 0;">📝 {qa["question"]}</div>', unsafe_allow_html=True)

        if qa['result']['found_in_knowledge_base']:
            st.markdown(f'<div class="answer-box">{qa["result"]["answer"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="answer-box" style="border-left-color:#9ca3af;">⚠ {qa["result"]["answer"]}</div>', unsafe_allow_html=True)

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
                           style="font-size:0.75rem;color:#c9a84c;text-decoration:none;white-space:nowrap;margin-left:8px;"
                           title="查看该文档所有检索片段">
                           📄 查看原文 →
                        </a>
                    </div>
                    <div class="citation-meta">
                        <span>📖 {loc_str}</span>
                        <span>🎯 相关度 {cite['score']:.4f}</span>
                    </div>
                    <div style="font-size:0.85rem;color:#4b5563;margin-top:0.4rem;line-height:1.6;">
                        {cite['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f'<span class="meta-tag">⏱ {qa["result"]["processing_time_ms"]:.0f} ms</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer-note">LexAI Legal Intelligence · 严谨 · 精确 · 可信</div>', unsafe_allow_html=True)
