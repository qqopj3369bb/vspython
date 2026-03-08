# backend/rag_engine.py
import os          #读取环境变量，操作文件路径
import uuid        #生成唯一文件id(给每个上传文档一个编号)
import re          #正则表达式，用于文本分割
import time        #延时，记时
from typing import List, Dict, Any, Optional   #给代码加类型提示，让代码更规范
from dotenv import load_dotenv                 #读取.env文件里的密钥
import httpx                                   #发送网络请求，调用API
import chromadb                                #本地向量数据库，存知识库向量

# 加载环境变量，读取项目根目录的 .env 文件，把里面的配置加载到系统里
load_dotenv()

# ============ 配置加载 ============
API_KEY = os.getenv("MINIMAX_API_KEY")
GROUP_ID = os.getenv("MINIMAX_GROUP_ID")
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "abab6.5s-chat")                           #使用的模型名称默认 abab6.5s
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"                                   #API 请求地址

# SiliconFlow 配置
SF_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SF_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-4B")       #读取（，默认），找不到用默认
SF_BASE_URL = "https://api.siliconflow.cn/v1"                                      #向量模型接口地址

#如果没填向量模型密钥，直接报错并停止运行
if not SF_API_KEY:                                                                 #if not:条件不成立/为空/没有时，执行后面语句
    raise ValueError("❌ 错误：未找到 SILICONFLOW_API_KEY 环境变量，请检查 .env 文件。")

#在控制台打印成功信息，f"..."让字符串能插入变量
print(f"✅ 配置加载成功 | LLM: {LLM_MODEL} | Embedding: {SF_MODEL_NAME}")


# ============ SiliconFlow Embedding 实现 ============
#一个文本向量生成类
class SiliconFlowEmbedding:
    """基于 SiliconFlow API 的高质量向量生成器"""
    
    #self必须写在第一个，str是类型，url有默认可传可不传参数，self = 这个类自己
    def __init__(self, api_key: str, model_name: str, base_url: str = "https://api.siliconflow.cn/v1"):
        #self.变量 = 外面的变量，整个类都能用
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.client = httpx.Client(timeout=60.0)  # 增加超时时间以防大文本
    
    #输入：一段 / 多段文本，输出：一堆数字（向量），返回 一组一组的浮点数列表，额外参数兼容其他调用方式
    def encode(self, texts: List[str], convert_to_numpy: bool = False, **kwargs) -> List[List[float]]:
        """
        调用 API 生成向量
        :param texts: 文本列表
        :param convert_to_numpy: 是否转换为 numpy 数组 (为了兼容旧接口，但这里主要返回 list)
        :return: 向量列表
        """
        if not texts:
            return []                                               
        #如果没有输入文本，直接返回空列表，避免报错

        headers = {                                       #headers=身份通行证+数据格式说明
            "Authorization": f"Bearer {self.api_key}",    #身份通行证
            "Content-Type": "application/json"            #数据格式说明
        }#带着 API KEY 去访问 API，证明有权限使用
        
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }#申请表：告诉接口：用哪个模型，要转向量的文本，返回浮点数格式的向量
        
        try:                                          #去寄快递
            response = self.client.post(              #交出包裹(给服务器发消息、提交任务)
                f"{self.base_url}/embeddings",        #地址
                json=payload,                         #里面的东西
                headers=headers                       #身份证
            )#向 SiliconFlow 发送请求，获取文本对应的向量
            
            if response.status_code == 200:             #HTTP 请求成功,服务器正常响应
                data = response.json()                   #把服务器返回的 JSON 数据，转成 Python 能看懂的字典
                # SiliconFlow 返回格式：data.data[i].embedding
                #从data里拿数组列表，把数据一条一条拿出来，每一条里每一个向量拿出来
                embeddings = [item["embedding"] for item in data["data"]]
                
                # 安全校验，确保顺序与输入一致 (API 通常保证顺序，但做个简单检查)
                if len(embeddings) != len(texts):          #几段文本就是几个向量
                    raise ValueError(f"返回向量数量 ({len(embeddings)}) 与输入文本数量 ({len(texts)}) 不匹配")
                
                if convert_to_numpy:                #如果需要 numpy 格式，就转成 numpy 数组
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

# 初始化全局 Embedding 实例，整个项目只创建一次，后面上传文档、提问检索都会用它生成向量
sf_embedder = SiliconFlowEmbedding(                   #给它命名
    api_key=SF_API_KEY,
    model_name=SF_MODEL_NAME,
    base_url=SF_BASE_URL
)
print(f"✅ SiliconFlow Embedding 初始化成功 (模型：{SF_MODEL_NAME})")


# ============ 文本分割器 (保持不变) ============
#不依赖第三方库，纯python实现
class SimpleTextSplitter:
    """纯 Python 文本分割器"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size                  #每个文本片段最多 500 个字符，防止太长导致向量不准
        self.chunk_overlap = chunk_overlap            #前后片段重叠 50 个字符，防止一句话被切断，保证上下文连贯
    
    #输入：一整篇长文本，输出：字符串列表（一堆小片段）
    def split_text(self, text: str) -> List[str]:
        """按段落和句子分割文本"""
        paragraphs = text.split('\n\n')               #先按 两个换行 分割成大段落
        chunks = []                                   #用空列表 chunks 存放最终切好的片段
        
        for para in paragraphs:                       #一段一段拿出来
            para = para.strip()                       # 去掉首尾空格、换行
            if not para:                              # 如果是空行，直接跳过
                continue
            
            if len(para) > self.chunk_size:
                # 简单的句子分割，长段落按句子分割
                sentences = re.split(r'(?<=[。！？!?])\s*', para)     #匹配句子结束符号，保证句子完整性
                current_chunk = ""                                   # 当前正在拼接的片段
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(current_chunk) + len(sentence) <= self.chunk_size:   ## 如果加上这句话不超过500，就继续加
                        current_chunk += sentence
                    else:                                             # 超过了，就保存当前片段，重新开始
                        if current_chunk:                             #有内容=true
                            chunks.append(current_chunk)              
                        current_chunk = sentence
                
                if current_chunk:                         #把最后剩下的内容加入列表
                    chunks.append(current_chunk)
            else:                                         #没有进行分割流程，短段落直接保存
                chunks.append(para)
        
        return chunks                                      #返回所有片段


# ============ RAG 引擎 ============

class RAGEngine:
    """RAG 核心引擎"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        # 自动定位数据库路径
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))     #找到当前所在文件夹
            persist_directory = os.path.join(base_dir, "storage", "chroma_db") #拼接最终保存路径
        
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)        #自动创建文件夹，无：创新的，存在：直接用
        
        try:                                                 #连接chromadb向量数据库
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(
                name="knowledge_base",                       # 知识库表名
                metadata={"hnsw:space": "cosine"}            #余弦相似度检索
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
            
            for i in range(0, len(texts), batch_size):         #按 20 个一组，循环遍历所有文本
                batch = texts[i:i+batch_size]                   #拿出当前这一组
                print(f"🔢 正在生成向量批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
                batch_emb = sf_embedder.encode(batch, convert_to_numpy=False)
                all_embeddings.extend(batch_emb)                 #放进总列表里
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
            return "⚠️ 未配置 MiniMax API,无法生成回答。请检查 .env 文件。"
        
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
            response = httpx.post(f"{url}?group_id={GROUP_ID}", json=payload, headers=headers, timeout=120)   #发送请求
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"错误：{response.text}"
        
        except Exception as e:
            return f"网络异常：{str(e)}"

    def upload_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """文档上传与知识库构建"""
        file_id = str(uuid.uuid4())            #生成唯一文件id
        temp_chunks = []
        temp_embeddings = []
        
        # 步骤 1：读取文件
        try:
            print(f"📄 正在读取文件：{filename}")
            with open(file_path, "r", encoding="utf-8") as f:     #安全打开文件，支持中文
                text = f.read()                                   #把文件里所有文字读出来，放进 text
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
        #传了id就在指定文件里，没有就全部

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=5 if file_id else 8,           #返回相似结果的条数
            include=["documents", "metadatas"],
            where=where_filter                       #只搜某个文件或搜全部
        )
        
        #找到的最相关的几段文字
        docs = results["documents"][0] if results["documents"] else []
        #提取找到的【文件信息】
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        
        if not docs:
            return {"answer": "知识库中未找到相关信息。", "sources": []}
        
        # 3. 构建上下文
        context_parts = []
        for i, (doc, meta) in enumerate(zip(docs, metadatas)):      #i = 片段序号（从 0 开始）
            doc_name = meta.get("filename", "未知文档")     #从信息里拿出文件名
            context_parts.append(f"[片段 {i+1}][来源：{doc_name}]\n{doc}")
        
        context = "\n\n".join(context_parts) #把所有片段拼在一起，用空行分开
        
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
            #在 AI 的回答里搜索固定格式的文字
            citation_match = re.search(r'引用片段 [:：]\s*([\d,\s,]+)', answer) 
            if citation_match:
                #拿到括号里匹配到的内容
                cited_indices = citation_match.group(1)
                #把字符串变成纯数字列表
                indices = [int(x.strip()) for x in re.findall(r'\d+', cited_indices)]
                
                used_docs = set()
                for idx in indices:                    #循环遍历 AI 引用的片段编号
                    if 1 <= idx <= len(metadatas):
                        #列表是从0开始的，所以要减1，metadatas[idx-1] → 拿到这个片段的信息
                        doc_name = metadatas[idx - 1].get("filename", "未知文档")
                        used_docs.add(doc_name)                  #把文件名放进集合里，多个片段来自同一个文件，只存一次
                
                actual_sources = list(used_docs)                #把集合转成普通列表
                # 清理回答中的引用行
                answer = re.sub(r'\n？引用片段 [:：]\s*[\d,\s,]+', '', answer).strip()
            else:
                #如果没找到引用行，直接取检索到的前 2 条参考资料的文件名
                actual_sources = list(set([m["filename"] for m in metadatas[:2]]))
        except Exception as e:
            print(f"⚠️ 解析引用失败：{e}")
            actual_sources = list(set([m["filename"] for m in metadatas[:2]]))
        
        return {"answer": answer, "sources": actual_sources}

    def get_document_list(self) -> List[Dict[str, Any]]:
        """获取文档列表"""
        all_metadata = self.collection.get(include=["metadatas"])
        file_map = {}
        
        #循环遍历每一个片段的信息
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
