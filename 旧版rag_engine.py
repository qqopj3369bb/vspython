# backend/rag_engine.py
import os
import uuid
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import httpx
import chromadb

# 加载环境变量
load_dotenv()

API_KEY = os.getenv("MINIMAX_API_KEY")
GROUP_ID = os.getenv("MINIMAX_GROUP_ID")
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "abab6.5s-chat")
BASE_URL = "https://api.minimax.chat/v1"


# ============ 简单的本地 Embedding（完全离线，无需下载模型）============
class SimpleLocalEmbedding:
    """基于词频的简单嵌入（完全离线，无需下载模型）"""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vocab = {}
        self.word_idx = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词：中文按字符，英文按单词"""
        words = []
        current_word = ""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)
            elif char.isalnum():
                current_word += char
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
        if current_word:
            words.append(current_word)
        return words
    
    def _get_embedding(self, text: str) -> List[float]:
        """生成固定维度向量"""
        words = self._tokenize(text.lower())
        vector = [0.0] * self.dim
        
        for i, word in enumerate(words):
            if word not in self.vocab:
                self.vocab[word] = self.word_idx % self.dim
                self.word_idx += 1
            idx = self.vocab[word]
            vector[idx] += 1.0
        
        # 归一化
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    def encode(self, texts: List[str], convert_to_numpy: bool = False, **kwargs):
        """兼容 sentence-transformers 接口"""
        embeddings = [self._get_embedding(text) for text in texts]
        if convert_to_numpy:
            import numpy as np
            return np.array(embeddings)
        return embeddings


# 初始化本地 Embedding（完全离线）
local_embedder = SimpleLocalEmbedding(dim=384)
print("✅ 本地 Embedding 初始化成功（离线模式）")


# ============ 文本分割器 ============
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
                sentences = para.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
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
    
    def __init__(self, persist_directory: str = "./backend/storage/chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        self.text_splitter = SimpleTextSplitter(chunk_size=500, chunk_overlap=50)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """使用本地 Embedding（完全离线）"""
        if not texts:
            print("⚠️ 警告：文本列表为空")
            return []
        
        try:
            embeddings = local_embedder.encode(texts, convert_to_numpy=False)
            print(f"✅ 本地生成 {len(embeddings)} 个向量（离线模式）")
            return embeddings
        except Exception as e:
            print(f"❌ 本地生成失败：{e}")
            return []

    def _call_minimax_chat(self, messages: List[Dict], prompt: str) -> str:
        """调用 MiniMax 聊天接口（仅用于生成回答）"""
        if not API_KEY or not GROUP_ID:
            return "⚠️ 未配置 MiniMax API，无法生成回答。请检查 .env 文件。"
        
        url = f"{BASE_URL}/text/chatcompletion_v2"
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
        """
        文档上传与知识库构建（带事务回滚机制）
        """
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
        
        # 步骤 3：生成向量（先不写入数据库）
        print("🔢 正在生成向量...")
        temp_embeddings = self._get_embeddings(temp_chunks)
        if not temp_embeddings:
            return {"status": "error", "message": "生成向量失败"}
        
        # 步骤 4：向量生成成功后再写入数据库（事务性操作）
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
        """
        知识库问答（优化参考来源显示）
        
        参数:
            question: 用户问题
            file_id: 可选，指定只检索某个文档（单文档检索模式）
        """
        print(f"❓ 收到问题：{question}")
        if file_id:
            print(f"📌 单文档检索模式：{file_id}")
        
        # 1. 问题向量化
        query_embedding = self._get_embeddings([question])
        if not query_embedding:
            return {"answer": "向量生成失败，请检查配置。", "sources": []}
        
        # 2. 检索相关片段
        where_filter = None
        if file_id:
            where_filter = {"file_id": file_id}
        
        # 检索更多片段，让 LLM 有选择空间
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=5 if file_id else 8,  # 全文档检索时获取更多片段
            include=["documents", "metadatas"],
            where=where_filter
        )
        
        docs = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        
        # 3. 构建带编号和来源的上下文
        if not docs:
            return {"answer": "知识库中未找到相关信息。", "sources": []}
        
        # 给每个片段添加编号和来源，方便 LLM 引用
        context_parts = []
        for i, (doc, meta) in enumerate(zip(docs, metadatas)):
            doc_name = meta.get("filename", "未知文档")
            context_parts.append(f"[片段 {i+1}][来源：{doc_name}]\n{doc}")
        
        context = "\n\n".join(context_parts)
        
        # 4. 构建 Prompt（要求 LLM 明确标注引用的片段编号）
        prompt = f"""你是一个基于知识库的智能助手。请根据以下【参考信息】回答用户的问题。

重要规则：
1. 【参考信息】中的每个片段都有编号，格式为 [片段 X][来源：文档名]
2. 请只使用与问题相关的片段来回答问题
3. 在回答的末尾，请单独一行列出你实际引用的片段编号，格式为：引用片段：1, 3, 5
4. 如果【参考信息】中没有答案，请直接说不知道，不要编造
5. 不要引用与问题无关的片段

【参考信息】:
{context}

用户问题：{question}
回答:"""
        
        messages = [{"role": "user", "content": prompt}]
        answer = self._call_minimax_chat(messages, prompt)
        
        # 5. 从回答中解析实际引用的片段编号
        actual_sources = []
        try:
            # 查找"引用片段："后面的内容
            citation_match = re.search(r'引用片段 [:：]\s*([\d,\s,]+)', answer)
            if citation_match:
                cited_indices = citation_match.group(1)
                # 解析片段编号
                indices = [int(x.strip()) for x in re.findall(r'\d+', cited_indices)]
                
                # 根据片段编号获取对应的文档名
                used_docs = set()
                for idx in indices:
                    if 1 <= idx <= len(metadatas):
                        doc_name = metadatas[idx - 1].get("filename", "未知文档")
                        used_docs.add(doc_name)
                
                actual_sources = list(used_docs)
                print(f"✅ 解析到引用片段：{indices}")
                print(f"✅ 实际参考来源：{actual_sources}")
                
                # 从回答中移除引用片段行（让回答更整洁）
                answer = re.sub(r'\n？引用片段 [:：]\s*[\d,\s,]+', '', answer).strip()
            else:
                # 如果没有找到引用片段标注，只使用前 2 个最相关的文档
                actual_sources = list(set([m["filename"] for m in metadatas[:2]]))
                print(f"⚠️ 未找到引用片段标注，使用前 2 个来源：{actual_sources}")
        except Exception as e:
            print(f"⚠️ 解析引用失败：{e}")
            actual_sources = list(set([m["filename"] for m in metadatas[:2]]))
        
        print(f"✅ 回答生成完成，参考来源：{actual_sources}")
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
