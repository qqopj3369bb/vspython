# frontend/app.py
import streamlit as st
import requests
import pandas as pd
import hashlib

import os  # 新增导入

# 【修改点】从环境变量读取 API URL，默认值为本地开发地址
# 在 Docker 中，docker-compose 会注入 API_URL=http://backend:8000/api
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api")

# 创建 requests session
session = requests.Session()

st.set_page_config(page_title="MiniMax RAG 知识库", layout="wide")
st.title("🤖 MiniMax 驱动的 RAG 知识库系统")

# ============ 初始化 Session State ============
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}  # {hash: filename}
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_doc_id' not in st.session_state:
    st.session_state.selected_doc_id = None

# ============ 辅助函数 ============
def get_file_hash(file_bytes):
    """计算文件哈希值，用于去重"""
    return hashlib.md5(file_bytes).hexdigest()

def refresh_document_list():
    """刷新文档列表"""
    try:
        docs_resp = session.get(f"{API_URL}/docs", timeout=10)
        return docs_resp.json() if docs_resp.status_code == 200 else []
    except Exception as e:
        st.error(f"无法连接后端服务：{e}")
        return []

def clear_uploaded_hash(filename):
    """删除文档时清理对应的哈希记录"""
    hash_to_remove = None
    for file_hash, fname in st.session_state.uploaded_files.items():
        if fname == filename:
            hash_to_remove = file_hash
            break
    if hash_to_remove:
        del st.session_state.uploaded_files[hash_to_remove]

# ============ 侧边栏：知识库管理 ============
with st.sidebar:
    st.header("📚 知识库管理")
    
    # 1. 上传文档（支持多文件选择）
    st.subheader("上传文档")
    
    # 关键：accept_multiple_files=True 允许一次选择多个文件
    uploaded_files = st.file_uploader(
        "选择文件 (txt/md)",
        type=['txt', 'md'],
        key="file_uploader",
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        st.info(f"📁 已选择 {len(uploaded_files)} 个文件")
        
        # 显示已选择的文件列表
        for i, f in enumerate(uploaded_files):
            st.caption(f"{i+1}. {f.name}")
        
        # 检查哪些文件已上传过
        files_to_upload = []
        already_uploaded = []
        
        for f in uploaded_files:
            file_hash = get_file_hash(f.getvalue())
            if file_hash in st.session_state.uploaded_files:
                already_uploaded.append(f.name)
            else:
                files_to_upload.append(f)
        
        # 提示已上传过的文件
        if already_uploaded:
            st.warning(f"⚠️ 以下文件已上传过，将跳过：{', '.join(already_uploaded)}")
        
        # 上传按钮
        if files_to_upload:
            if st.button(f"📤 上传 {len(files_to_upload)} 个文件", key="upload_btn"):
                success_count = 0
                fail_count = 0
                
                progress_bar = st.progress(0)
                
                for i, f in enumerate(files_to_upload):
                    with st.spinner(f"正在上传：{f.name}..."):
                        try:
                            resp = session.post(
                                f"{API_URL}/upload",
                                files={"file": f},
                                timeout=300
                            )
                            if resp.status_code == 200:
                                result = resp.json()
                                st.success(f"✅ {f.name}: 切片数 {result.get('chunk_count')}")
                                # 记录已上传文件
                                file_hash = get_file_hash(f.getvalue())
                                st.session_state.uploaded_files[file_hash] = f.name
                                success_count += 1
                            else:
                                st.error(f"❌ {f.name}: {resp.text}")
                                fail_count += 1
                        except Exception as e:
                            st.error(f"❌ {f.name}: {str(e)}")
                            fail_count += 1
                    
                    # 更新进度条
                    progress_bar.progress((i + 1) / len(files_to_upload))
                
                st.divider()
                st.info(f"📊 上传完成：成功 {success_count} 个，失败 {fail_count} 个")
    
    st.divider()
    
    # 2. 刷新按钮
    if st.button("🔄 刷新文档列表", key="refresh_btn"):
        st.rerun()
    
    # 3. 文档列表与删除
    st.subheader("已存文档")
    docs = refresh_document_list()
    
    if docs:
        df = pd.DataFrame(docs)
        st.dataframe(df[["filename", "chunk_count"]], use_container_width=True)
        st.caption(f"共 {len(docs)} 个文档")
        
        # 文档选择（用于单文档检索）
        st.subheader("📌 检索设置")
        search_mode = st.radio(
            "检索模式",
            ["🔍 全部文档检索", "📄 单文档检索"],
            key="search_mode"
        )
        
        selected_doc_id = None
        if search_mode == "📄 单文档检索":
            selected_doc_id = st.selectbox(
                "选择要检索的文档",
                [d["file_id"] for d in docs],
                format_func=lambda x: next((d["filename"] for d in docs if d["file_id"]==x), x),
                key="search_doc_select"
            )
            st.session_state.selected_doc_id = selected_doc_id
            doc_name = next((d['filename'] for d in docs if d['file_id']==selected_doc_id), '未知')
            st.info(f"当前只检索：{doc_name}")
        else:
            st.session_state.selected_doc_id = None
        
        # 删除功能
        st.subheader("🗑️ 删除文档")
        delete_id = st.selectbox(
            "选择要删除的文档",
            [d["file_id"] for d in docs],
            format_func=lambda x: next((d["filename"] for d in docs if d["file_id"]==x), x),
            key="delete_select"
        )
        if st.button("删除选中文档", key="delete_btn"):
            delete_filename = next((d["filename"] for d in docs if d["file_id"]==delete_id), None)
            
            resp = session.delete(f"{API_URL}/docs/{delete_id}", timeout=10)
            if resp.status_code == 200:
                st.success("✅ 删除成功")
                if delete_filename:
                    clear_uploaded_hash(delete_filename)
                st.rerun()
            else:
                st.error("❌ 删除失败")
        
        # 查看片段功能
        st.subheader("📄 查看解析片段")
        view_id = st.selectbox(
            "选择查看片段的文档",
            [d["file_id"] for d in docs],
            key="view_sel",
            format_func=lambda x: next((d["filename"] for d in docs if d["file_id"]==x), x)
        )
        if st.button("加载片段", key="load_chunks_btn"):
            resp = session.get(f"{API_URL}/docs/{view_id}/chunks", timeout=10)
            if resp.status_code == 200:
                chunks = resp.json()
                st.info(f"共 {len(chunks)} 个片段")
                for chunk in chunks:
                    with st.expander(f"片段 {chunk['index']}"):
                        st.text(chunk['content'])
            else:
                st.error("❌ 加载失败")
    else:
        st.info("📭 暂无文档，请上传。")

# ============ 主区域：问答交互 ============
st.header("💬 知识库问答")

# 显示检索模式提示
if st.session_state.selected_doc_id:
    docs = refresh_document_list()
    doc_name = next((d["filename"] for d in docs if d["file_id"]==st.session_state.selected_doc_id), "未知")
    st.info(f"📌 当前仅检索文档：**{doc_name}**")

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📎 参考来源：{', '.join(msg['sources'])}")

# 用户输入
user_question = st.chat_input("请输入关于文档内容的问题...")

if user_question:
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 正在检索并生成答案..."):
            try:
                payload = {"question": user_question}
                if st.session_state.selected_doc_id:
                    payload["file_id"] = st.session_state.selected_doc_id
                
                resp = session.post(
                    f"{API_URL}/query",
                    data=payload,
                    timeout=120
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.write(data["answer"])
                    if data["sources"]:
                        st.caption(f"📎 参考来源：{', '.join(data['sources'])}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources"]
                    })
                else:
                    st.error(f"❌ 请求失败：{resp.text}")
            except Exception as e:
                st.error(f"❌ 连接错误：{str(e)}")
