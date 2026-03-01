# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖 (编译某些 Python 包需要)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
# 注意：这里我们使用根目录的 requirements.txt，因为它包含了所有依赖
# 如果 backend 有独立的 requirements.txt，请改为 COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
