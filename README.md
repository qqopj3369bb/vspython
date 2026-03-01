# MiniMax-RAG 知识库系统

基于 MiniMax LLM 与 Qwen Embedding 的智能知识库问答系统。支持本地向量检索、多文档混合搜索及容器化部署。

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.32-red.svg)](https://streamlit.io/)

</div>

## ✨ 项目特点

- 🔒 **数据安全**：本地向量库存储，无敏感数据上云（仅模型推理接口）。
- 🧠 **智能检索**：接入 SiliconFlow Qwen 大模型进行语义向量化，精准匹配用户问题。
- 💬 **多模态对话**：基于 MiniMax 大语言模型生成流畅自然的答案。
- 🐳 **容器化部署**：支持 Docker/Docker-compose 一键拉起前后端服务。
- 🌍 **多平台托管**：代码已同步至 GitHub/GitLab/Gitee/Codeberg，便于团队协作。

## 🚀 快速开始

### 方式一：本地开发运行 (推荐调试)

1.  **克隆代码**
    ```bash
    git clone https://github.com/youruser/vspyton.git
    cd vspyton/backend
    ```

2.  **配置环境变量**
    复制 `.env.example` 为 `.env` 并填入你的 API Key。
    ```bash
    cp ../.env.example ../.env
    # 编辑 .env 填入 MINIMAX_API_KEY, SILICONFLOW_API_KEY 等
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

4.  **启动服务**
    ```bash
    # 终端 1：启动后端
    uvicorn main:app --reload --port 8000

    # 终端 2：启动前端
    streamlit run ../frontend/app.py
    ```
    访问 `http://localhost:8501`

### 方式二：Docker 容器化部署 (生产环境)

1.  **准备镜像与环境**
    确保已安装 [Docker](https://www.docker.com/) 和 [Docker Compose](https://docs.docker.com/compose/install/)。

2.  **配置密钥**
    在 `vspython` 根目录创建 `.env` 文件：
    ```bash
    cat > .env << EOF
    MINIMAX_API_KEY=sk-your-key
    MINIMAX_GROUP_ID=your-group-id
    SILICONFLOW_API_KEY=sk-your-sf-key
    EMBEDDING_MODEL_NAME=Qwen/Qwen2.5-Embedding-72B
    EOF
    ```

3.  **启动容器**
    ```bash
    docker-compose up -d --build
    ```
    此时后端 API 暴露于 `http://localhost:8000`，前端 UI 暴露于 `http://localhost:8501`。

4.  **查看日志**
    ```bash
    docker-compose logs -f
    ```

5.  **停止服务**
    ```bash
    docker-compose down
    ```
    *(注：数据已持久化在 `./backend/storage` 卷中，删除容器不会丢失知识库)*

## 📂 目录说明

- `backend/`: FastAPI 后端服务，包含 RAG 核心引擎、API 接口及向量库操作。
- `frontend/`: Streamlit 前端交互界面，提供文件上传、问答会话功能。
- `storage/`: ChromaDB 向量数据库及上传文件的物理存储路径。
- `Dockerfile`: 后端服务 Docker 构建文件。
- `docker-compose.yml`: 容器编排配置。

## ⚠️ 注意事项

- **内存占用**：首次加载 Qwen 向量模型时可能需要较大显存/内存，请确保环境充足。
- **API Key**：本项目仅涉及模型 API 调用，不包含模型本体训练费用。
- **数据备份**：建议定期备份 `backend/storage` 目录以防数据丢失。

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request。请务必遵守 [Conventional Commits](https://www.conventionalcommits.org/) 规范撰写 Commit Message。

1. Fork 本仓库。
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)。
3. 提交更改 (`git commit -am 'Add some AmazingFeature'`)。
4. 推送到分支 (`git push origin feature/AmazingFeature`)。
5. 开启 Pull Request。

## 📜 许可证

MIT License

---
**Created by ZXK | Powered by MiniMax & SiliconFlow**
