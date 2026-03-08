# backend/main.py
import os
import sys                  

# 添加当前文件所在目录的父目录到 Python 路径
#获取文件的绝对路径，获取文件所在的文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))
#把当前文件夹加入 Python 的 “模块搜索路径”
sys.path.insert(0, current_dir)

from fastapi import FastAPI, UploadFile, File, HTTPException, Form     #fastapi创建web服务，UploadFile, File接受上传文件
from fastapi.middleware.cors import CORSMiddleware                    #HTTPException抛出接口异常，Form接受表单数据提问用
import shutil                                                         #CORSMiddleware解决跨域问题，shutil保存上传文件
from rag_engine import engine                                          #导入写好的 RAG 核心引擎

app = FastAPI(title="MiniMax RAG API")                        #创建一个fastapi实例，整个后端服务的根对象

# 允许跨域
app.add_middleware(
    CORSMiddleware,                                  #端口不同 → 浏览器默认禁止访问，加上这段代码 → 解除禁止，允许访问
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)                                                     #允许所有域名、所有请求、所有请求头访问后端,解决前端访问后端的跨域问题

UPLOAD_DIR = "./backend/storage/uploads"             #设置上传文件保存路径 
os.makedirs(UPLOAD_DIR, exist_ok=True)               #自动创建目录

@app.post("/api/upload")                              #创建一个 POST 接口
async def upload_document(file: UploadFile = File(...)):         #定义异步上传函数
    """上传文档并构建知识库"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)       #拼接
    try:
        #保存文件到本地
        with open(file_path, "wb") as buffer:            #以二进制方式打开文件
            shutil.copyfileobj(file.file, buffer)        #把前端上传的文件，写入本地硬盘
        
        #调用RAG引擎处理文档
        result = engine.upload_document(file_path, file.filename)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_knowledge(question: str = Form(...), file_id: str = Form(None)):
    """
    知识库问答
    
    参数:
        question: 用户问题（必需）
        file_id: 指定检索的文档 ID(可选,用于单文档检索模式)
    """
    try:
        return engine.query_knowledge(question, file_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/docs")
async def list_documents():
    """获取所有文档列表"""
    return engine.get_document_list()         #把结果直接返回给前端

@app.delete("/api/docs/{file_id}")
async def delete_document(file_id: str):
    """删除指定文档"""
    success = engine.delete_document(file_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "file_id": file_id}

@app.get("/api/docs/{file_id}/chunks")
async def get_chunks(file_id: str):
    """查看文档解析后的文本片段"""
    return engine.get_document_chunks(file_id)

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "message": "RAG API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
