import os
import uuid
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import httpx
import chromadb
import json
import time

# -------------------------- 环境变量加载 --------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, '.env')
load_dotenv(dotenv_path=env_path)

# -------------------------- 全局配置 --------------------------
API_KEY = os.getenv("MINIMAX_API_KEY")
GROUP_ID = os.getenv("MINIMAX_GROUP_ID")
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "abab6.5s-chat")
BASE_URL = "https://api.minimax.chat/v1"

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "").strip()
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1/embeddings"
SILICONFLOW_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen2.5-Embedding-72B") 

print(f"🚀 RAG Engine Start | Model: {SILICONFLOW_MODEL}")
if not SILICONFLOW_API_KEY: raise ValueError("❌ SiliconFlow Key missing in .env")

# -------------------------- 文本分割器 --------------------------
class SimpleTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size; self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        paragraphs = text.split('\n\n')
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para: continue
            if len(para) > self.chunk_size:
                sentences = para.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
                current_chunk = ""
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence: continue
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk: chunks.append(current_chunk)
                        current_chunk = sentence
                if current_chunk: chunks.append(current_chunk)
            else:
                chunks.append(para)
        return chunks

# -------------------------- RAG 引擎 --------------------------
class RAGEngine:
    def __init__(self, persist_directory: str = "./backend/storage"):
        root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "storage")
        self.persist_directory = root_dir
        
        os.makedirs(root_dir, exist_ok=True)
        # 强制指定 cosine 空间
        self.client = chromadb.PersistentClient(path=root_dir)
        self.collection = self.client.get_or_create_collection(name="knowledge_base", metadata={"hnsw:space": "cosine"})
        self.text_splitter = SimpleTextSplitter(chunk_size=500, chunk_overlap=50)
        
        # 【修复】降低阈值，适应高质量模型
        self.distance_threshold = 0.65 
        
        print(f"✅ SiliconFlow Embedding Connected | Threshold: {self.distance_threshold}")

    def _call_siliconflow_embedding(self, texts: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": SILICONFLOW_MODEL, "input": texts, "encoding_format": "float"}
        try:
            response = httpx.post(SILICONFLOW_BASE_URL, json=payload, headers=headers, timeout=60)
            if response.status_code == 200:
                data = response.json()
                results = [item["embedding"] for item in data.get("data", [])]
                print(f"✅ 向量化成功：{len(results)} 条")
                return results
            elif response.status_code == 429:
                time.sleep(2); return self._call_siliconflow_embedding(texts)
            else:
                print(f"❌ API Error: {response.text}")
                return []
        except Exception as e:
            print(f"❌ Network Error: {str(e)}"); return []

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._call_siliconflow_embedding(texts) if texts else []

    def _call_minimax_chat(self, messages: List[Dict], prompt: str) -> str:
        url = f"{BASE_URL}/text/chatcompletion_v2"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        payload = {"model": LLM_MODEL, "messages": messages, "stream": False}
        try:
            resp = httpx.post(f"{url}?group_id={GROUP_ID}", json=payload, headers=headers, timeout=120)
            if resp.status_code == 200: return resp.json()["choices"][0]["message"]["content"]
            else: return f"错误：{resp.text}"
        except Exception as e: return f"网络异常：{str(e)}"

    def upload_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        file_id = str(uuid.uuid4())
        temp_chunks = []
        try:
            with open(file_path, "r", encoding="utf-8") as f: 
                text = f.read()
        except Exception as e: 
            return {"status": "error", "message": str(e)}
        if not text.strip(): 
            return {"status": "error", "message": "内容为空"}
        temp_chunks = self.text_splitter.split_text(text)
        chunk_count = len(temp_chunks)
        if not temp_chunks: 
            return {"status": "error", "message": "分片失败"}
        
        embeddings = self._get_embeddings(temp_chunks)
        if not embeddings: 
            return {"status": "error", "message": "生成向量失败"}
        
        try:
            metadatas = [{"file_id": file_id, "filename": filename} for _ in temp_chunks]
            ids = [f"{file_id}_{i}" for i in range(len(temp_chunks))]
            self.collection.add(documents=temp_chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
            print(f"✅ 入库成功：{filename} ({chunk_count})")
            return {"status": "success", "chunk_count": chunk_count, "name": filename}
        except Exception as e: 
            return {"status": "error", "message": str(e)}

    def query_knowledge(self, question: str, file_id: Optional[str] = None) -> Dict[str, Any]:
        """【核心修复】增加完整异常捕获"""
        print(f"❓ Question: {question}")
        try:
            # 1. 基础检查
            count = self.collection.count()
            if count == 0:
                return {"answer": "数据库为空，请先上传文档。", "sources": []}
                
            q_emb = self._get_embeddings([question])
            if not q_emb: 
                return {"answer": "向量化失败", "sources": []}
            
            w_filter = {"file_id": file_id} if file_id else None
            
            # 2. 检索 (必须包含 distances)
            res = self.collection.query(query_embeddings=q_emb, n_results=20, include=["documents", "metadatas", "distances"], where=w_filter)
            
            docs = res["documents"][0] if res.get("documents") else []
            metas = res["metadatas"][0] if res.get("metadatas") else []
            dists = res["distances"][0] if res.get("distances") else []
            
            # 【修复 1】安全获取最小值
            min_dist_val = min(dists) if dists else 1.0
            print(f"📊 检索数量：{len(docs)}, 最小距离：{min_dist_val:.3f}")
            
            # 3. 严格过滤逻辑 (核心修复)
            valid_docs = []
            valid_metas = []
            
            for doc, meta, dist in zip(docs, metas, dists):
                # 条件 1：距离必须符合阈值
                is_dist_ok = dist <= self.distance_threshold
                
                # 条件 2：文件名必须真实存在且非默认值
                name = meta.get("filename", "")
                is_name_valid = bool(name and name != "未知文档")
                
                if is_dist_ok and is_name_valid:
                    valid_docs.append(doc)
                    valid_metas.append(meta)
                    
            print(f"✅ 高可信片段：{len(valid_docs)} / {len(docs)}")
            
            # 4. 兜底策略
            if not valid_docs:
                if len(docs) > 0 and min(dists) < self.distance_threshold * 1.5:
                     # 若勉强接近，尝试最相关的
                     idx = dists.index(min(dists))
                     valid_docs = [docs[idx]]; valid_metas = [metas[idx]]
                     print("⚠️ 采用宽松模式 Top1")
                else:
                     return {"answer": "未找到相关内容。", "sources": []}
                     
            # 5. 构建最终上下文
            parts = []
            final_sources = set()
            for i, (doc, meta) in enumerate(zip(valid_docs, valid_metas)):
                name = meta.get("filename")
                final_sources.add(name)
                parts.append(f"[片段 {i+1}][来源：{name}]\n{doc}")
                
            context = "\n\n".join(parts)
            prompt = f"请参考以下内容回答：\n{context}\n问题：{question}"
            
            answer = self._call_minimax_chat([{"role": "user", "content": prompt}], prompt)
            return {"answer": answer, "sources": list(final_sources)}
            
        except Exception as e:
            print(f"❌ Query Error: {e}")
            import traceback
            traceback.print_exc()
            return {"answer": f"系统错误：{str(e)}", "sources": []}

    def get_document_list(self) -> List[Dict[str, Any]]:
        all_meta = self.collection.get(include=["metadatas"])
        f_map = {}
        for m in all_meta.get("metadatas", []):
            fid = m["file_id"]
            if fid not in f_map: 
                f_map[fid] = {"file_id": fid, "filename": m.get("filename")}
            f_map[fid]["chunk_count"] = f_map[fid].get("chunk_count", 0) + 1
        return list(f_map.values())
    
    def delete_document(self, file_id: str) -> bool:
        try:
            self.collection.delete(where={"file_id": file_id})
            return True
        except: 
            return False

    def get_document_chunks(self, file_id: str) -> List[Dict[str, Any]]:
        res = self.collection.get(where={"file_id": file_id}, include=["documents"])
        return [{"index": i, "content": d[:200]+("..." if len(d)>200 else "")} for i, d in enumerate(res.get("documents", []))]

engine = RAGEngine()
