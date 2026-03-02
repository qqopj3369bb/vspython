# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# 【修改点】直接复制当前目录 (backend) 下的 requirements.txt
# 因为 docker-compose 中 context 设为 ./backend，所以这里的 . 就是 backend 文件夹
COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
