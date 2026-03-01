# 使用官方 Python 精简版基础镜像
FROM python:3.10-slim as builder

# 设置工作目录
WORKDIR /app

# 安装编译依赖 (chromadb 可能需要一些编译工具)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 第二阶段：生产环境（更小）
FROM python:3.10-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 从 builder 阶段复制已安装的包
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 创建非 root 用户以增强安全性 (可选，开发可跳过)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 健康检查 (可选)
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/api/health || exit 1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
