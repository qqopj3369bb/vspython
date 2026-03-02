# frontend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# 【修改点】直接复制当前目录 (frontend) 下的 requirements.txt
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 复制前端代码
COPY . .

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 启动命令
# server.address 设为 0.0.0.0 允许外部访问
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
