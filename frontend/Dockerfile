# 使用官方 Python 精简版基础镜像
FROM python:3.10-slim

# 设置环境参数，避免生成 .pyc 文件并防止输出缓冲
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

# 创建工作目录
WORKDIR /app

# 复制依赖文件以利用 Docker 层缓存（如果只修改代码不修改依赖，不会重新安装）
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制当前应用代码（仅复制 frontend 内容）
COPY . .

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 启动命令
# --server.headless=true：允许在容器无头模式下运行
# --server.address=0.0.0.0：允许外部访问
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true", "--server.address=0.0.0.0"]
