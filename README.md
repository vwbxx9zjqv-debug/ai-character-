# 🎀 Virtual Character Companion

AI 虚拟二次元角色伴侣 — 可扩展的 AI 伴侣系统。

## 项目概述

一个能独立运行的 AI 虚拟角色设备（ESP32 硬件），具有完整的语音对话能力、情感表达和长期记忆。用户可自上传角色模型和声音模型。

### 技术栈

- **后端**: Python 3.11+ / FastAPI / WebSocket / 插件架构
- **前端**: ESP32 固件 (ESP-IDF + LVGL) + Web 管理后台
- **AI**: Claude / OpenAI / Whisper / Edge-TTS / CosyVoice / GPT-SoVITS
- **硬件**: ESP32-2432S028R (含蓝牙 A2DP)

### 一期功能

- 免按键语音对话 (VAD 自动检测)
- 分层角色动画 (情绪表情 + 嘴型同步)
- 蓝牙耳机连接
- 角色模型自上传 + 切换
- 声音模型自上传 + 切换
- Web 管理后台
- 三个内置二次元角色

### 开源项目复用

| 项目 | 用途 | 复用方式 |
|------|------|---------|
| [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) | Provider 架构 + Live2D 管线 | 参考架构 + 适配 Provider |
| [easy-live2d](https://github.com/Panzer-Jack/easy-live2d) | Web Live2D 角色渲染 | 集成 (Phase 3) |
| [fastapi-amis-admin](https://github.com/amisadmin/fastapi-amis-admin) | 管理后台骨架 | 可选集成 |
| [SoulSpeak](https://github.com/chengzi0103/SoulSpeak) | 情感对话参考 | 参考架构 |

## 快速开始

```bash
cd backend

# 安装依赖
pip install -e .

# 配置 API Keys
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_API_KEY 和 OPENAI_API_KEY

# 启动开发服务器
uvicorn main:app --reload --port 8000
```

然后访问：
- 管理后台: http://localhost:8000/static/admin.html
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## 项目结构

```
backend/
├── core/              # 核心框架 (事件总线/插件系统/WS协议)
├── plugins/           # Provider 插件 (STT/LLM/TTS)
├── routes/            # API 路由 + WebSocket
├── services/          # 业务逻辑 (对话编排)
├── db/                # 数据库 (Schema + Queries)
├── static/            # Web 管理后台
└── main.py            # 入口
```

## 开发路线

- P0: 基础框架 ✅ (当前)
- P1: 核心对话链路
- P2: 角色动画系统
- P3: 人设系统 + 长期记忆
- P4: 自上传系统
- P5: 情感语音 (CosyVoice 2)
- P6: 联调打磨
