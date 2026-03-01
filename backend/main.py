# backend/main.py
import os
import sys

# 添加当前文件所在目录的父目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
from rag_engine import engine

app = FastAPI(title="MiniMax RAG API")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./backend/storage/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档并构建知识库"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
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
        file_id: 指定检索的文档 ID（可选，用于单文档检索模式）
    """
    try:
        return engine.query_knowledge(question, file_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/docs")
async def list_documents():
    """获取所有文档列表"""
    return engine.get_document_list()

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
