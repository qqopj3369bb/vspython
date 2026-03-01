# backend/rag_engine.py
import os
import uuid
import re
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import httpx
import chromadb

# 加载环境变量
load_dotenv()

# ============ 配置加载 ============
API_KEY = os.getenv("MINIMAX_API_KEY")
GROUP_ID = os.getenv("MINIMAX_GROUP_ID")
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "abab6.5s-chat")
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"

# SiliconFlow 配置
SF_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SF_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-4B")
SF_BASE_URL = "https://api.siliconflow.cn/v1"

if not SF_API_KEY:
    raise ValueError("❌ 错误：未找到 SILICONFLOW_API_KEY 环境变量，请检查 .env 文件。")

print(f"✅ 配置加载成功 | LLM: {LLM_MODEL} | Embedding: {SF_MODEL_NAME}")


# ============ SiliconFlow Embedding 实现 ============
class SiliconFlowEmbedding:
    """基于 SiliconFlow API 的高质量向量生成器"""
    
    def __init__(self, api_key: str, model_name: str, base_url: str = "https://api.siliconflow.cn/v1"):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.client = httpx.Client(timeout=60.0)  # 增加超时时间以防大文本
    
    def encode(self, texts: List[str], convert_to_numpy: bool = False, **kwargs) -> List[List[float]]:
        """
        调用 API 生成向量
        :param texts: 文本列表
        :param convert_to_numpy: 是否转换为 numpy 数组 (为了兼容旧接口，但这里主要返回 list)
        :return: 向量列表
        """
        if not texts:
            return []
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                # SiliconFlow 返回格式：data.data[i].embedding
                embeddings = [item["embedding"] for item in data["data"]]
                
                # 确保顺序与输入一致 (API 通常保证顺序，但做个简单检查)
                if len(embeddings) != len(texts):
                    raise ValueError(f"返回向量数量 ({len(embeddings)}) 与输入文本数量 ({len(texts)}) 不匹配")
                
                if convert_to_numpy:
                    import numpy as np
                    return np.array(embeddings)
                
                return embeddings
            
            else:
                error_msg = f"SiliconFlow API 错误 [{response.status_code}]: {response.text}"
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
                
        except httpx.RequestError as e:
            raise Exception(f"网络请求失败：{str(e)}")
        except Exception as e:
            raise Exception(f"向量生成异常：{str(e)}")

# 初始化全局 Embedding 实例
sf_embedder = SiliconFlowEmbedding(
    api_key=SF_API_KEY,
    model_name=SF_MODEL_NAME,
    base_url=SF_BASE_URL
)
print(f"✅ SiliconFlow Embedding 初始化成功 (模型：{SF_MODEL_NAME})")


# ============ 文本分割器 (保持不变) ============
class SimpleTextSplitter:
    """纯 Python 文本分割器"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """按段落和句子分割文本"""
        paragraphs = text.split('\n\n')
        chunks = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(para) > self.chunk_size:
                # 简单的句子分割
                sentences = re.split(r'(?<=[。！？!?])\s*', para)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk)
            else:
                chunks.append(para)
        
        return chunks


# ============ RAG 引擎 ============
class RAGEngine:
    """RAG 核心引擎"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        # 自动定位数据库路径
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            persist_directory = os.path.join(base_dir, "storage", "chroma_db")
        
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ ChromaDB 初始化成功：{persist_directory}")
        except Exception as e:
            print(f"❌ ChromaDB 初始化失败：{e}")
            raise e
        
        self.text_splitter = SimpleTextSplitter(chunk_size=500, chunk_overlap=50)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """使用 SiliconFlow API 生成向量"""
        if not texts:
            print("⚠️ 警告：文本列表为空")
            return []
        
        try:
            # 分批处理，避免单次请求过大 (SiliconFlow 通常支持较大 batch，但保守起见分片)
            batch_size = 20
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                print(f"🔢 正在生成向量批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
                batch_emb = sf_embedder.encode(batch, convert_to_numpy=False)
                all_embeddings.extend(batch_emb)
                # 可选：短暂休眠避免触发频率限制
                # time.sleep(0.1) 
            
            print(f"✅ 成功生成 {len(all_embeddings)} 个向量 (SiliconFlow)")
            return all_embeddings
            
        except Exception as e:
            print(f"❌ 向量生成失败：{e}")
            return []

    def _call_minimax_chat(self, messages: List[Dict], prompt: str) -> str:
        """调用 MiniMax 聊天接口"""
        if not API_KEY or not GROUP_ID:
            return "⚠️ 未配置 MiniMax API，无法生成回答。请检查 .env 文件。"
        
        url = f"{MINIMAX_BASE_URL}/text/chatcompletion_v2"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False
        }
        
        try:
            response = httpx.post(f"{url}?group_id={GROUP_ID}", json=payload, headers=headers, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"错误：{response.text}"
        
        except Exception as e:
            return f"网络异常：{str(e)}"

    def upload_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """文档上传与知识库构建"""
        file_id = str(uuid.uuid4())
        temp_chunks = []
        temp_embeddings = []
        
        # 步骤 1：读取文件
        try:
            print(f"📄 正在读取文件：{filename}")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            return {"status": "error", "message": f"读取文件失败：{str(e)}"}
        
        # 步骤 2：文本分割
        temp_chunks = self.text_splitter.split_text(text)
        if not temp_chunks:
            return {"status": "error", "message": "文档内容为空"}
        print(f"✂️ 文本分割完成：{len(temp_chunks)} 个片段")
        
        # 步骤 3：生成向量 (调用 SiliconFlow)
        print("🔢 正在调用 SiliconFlow 生成向量...")
        temp_embeddings = self._get_embeddings(temp_chunks)
        if not temp_embeddings or len(temp_embeddings) != len(temp_chunks):
            return {"status": "error", "message": "生成向量失败，请检查 API Key 或网络连接"}
        
        # 步骤 4：写入数据库
        try:
            print("💾 正在写入向量库...")
            self.collection.add(
                documents=temp_chunks,
                embeddings=temp_embeddings,
                metadatas=[
                    {"file_id": file_id, "filename": filename, "chunk_index": i}
                    for i in range(len(temp_chunks))
                ],
                ids=[f"{file_id}_{i}" for i in range(len(temp_chunks))]
            )
            print(f"✅ 文档上传成功：{filename} ({len(temp_chunks)} 个片段)")
            return {
                "status": "success",
                "file_id": file_id,
                "filename": filename,
                "chunk_count": len(temp_chunks)
            }
        except Exception as e:
            print(f"❌ 写入数据库失败：{e}")
            return {"status": "error", "message": f"写入数据库失败：{str(e)}"}

    def query_knowledge(self, question: str, file_id: Optional[str] = None) -> Dict[str, Any]:
        """知识库问答"""
        print(f"❓ 收到问题：{question}")
        
        # 1. 问题向量化
        query_embedding = self._get_embeddings([question])
        if not query_embedding:
            return {"answer": "向量生成失败，请检查 SiliconFlow 配置。", "sources": []}
        
        # 2. 检索相关片段
        where_filter = {"file_id": file_id} if file_id else None
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=5 if file_id else 8,
            include=["documents", "metadatas"],
            where=where_filter
        )
        
        docs = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        
        if not docs:
            return {"answer": "知识库中未找到相关信息。", "sources": []}
        
        # 3. 构建上下文
        context_parts = []
        for i, (doc, meta) in enumerate(zip(docs, metadatas)):
            doc_name = meta.get("filename", "未知文档")
            context_parts.append(f"[片段 {i+1}][来源：{doc_name}]\n{doc}")
        
        context = "\n\n".join(context_parts)
        
        # 4. 构建 Prompt
        prompt = f"""你是一个基于知识库的智能助手。请根据以下【参考信息】回答用户的问题。

重要规则：
1. 【参考信息】中的每个片段都有编号，格式为 [片段 X][来源：文档名]
2. 请只使用与问题相关的片段来回答问题
3. 在回答的末尾，请单独一行列出你实际引用的片段编号，格式为：引用片段：1, 3, 5
4. 如果【参考信息】中没有答案，请直接说不知道，不要编造

【参考信息】:
{context}

用户问题：{question}
回答:"""
        
        messages = [{"role": "user", "content": prompt}]
        answer = self._call_minimax_chat(messages, prompt)
        
        # 5. 解析引用来源
        actual_sources = []
        try:
            citation_match = re.search(r'引用片段 [:：]\s*([\d,\s,]+)', answer)
            if citation_match:
                cited_indices = citation_match.group(1)
                indices = [int(x.strip()) for x in re.findall(r'\d+', cited_indices)]
                
                used_docs = set()
                for idx in indices:
                    if 1 <= idx <= len(metadatas):
                        doc_name = metadatas[idx - 1].get("filename", "未知文档")
                        used_docs.add(doc_name)
                
                actual_sources = list(used_docs)
                # 清理回答中的引用行
                answer = re.sub(r'\n？引用片段 [:：]\s*[\d,\s,]+', '', answer).strip()
            else:
                actual_sources = list(set([m["filename"] for m in metadatas[:2]]))
        except Exception as e:
            print(f"⚠️ 解析引用失败：{e}")
            actual_sources = list(set([m["filename"] for m in metadatas[:2]]))
        
        return {"answer": answer, "sources": actual_sources}

    def get_document_list(self) -> List[Dict[str, Any]]:
        """获取文档列表"""
        all_metadata = self.collection.get(include=["metadatas"])
        file_map = {}
        
        for meta in all_metadata["metadatas"]:
            fid = meta["file_id"]
            if fid not in file_map:
                file_map[fid] = {
                    "file_id": fid,
                    "filename": meta["filename"],
                    "chunk_count": 0
                }
            file_map[fid]["chunk_count"] += 1
        
        return list(file_map.values())

    def delete_document(self, file_id: str) -> bool:
        """删除文档"""
        try:
            self.collection.delete(where={"file_id": file_id})
            print(f"🗑️ 文档删除成功：{file_id}")
            return True
        except Exception as e:
            print(f"❌ 删除失败：{e}")
            return False

    def get_document_chunks(self, file_id: str) -> List[Dict[str, Any]]:
        """查看文档片段"""
        results = self.collection.get(
            where={"file_id": file_id},
            include=["documents"]
        )
        
        chunks = []
        for i, doc in enumerate(results["documents"]):
            chunks.append({
                "index": i,
                "content": doc[:200] + "..." if len(doc) > 200 else doc
            })
        
        return chunks


# ============ 单例实例 ============
engine = RAGEngine()
