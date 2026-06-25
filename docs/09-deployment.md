# 09 — 部署与运维

## 部署架构

```
                         ┌──────────────────┐
                         │    User Browser   │
                         └────────┬─────────┘
                                  │ HTTPS
                         ┌────────┴─────────┐
                         │  nginx (:443)     │
                         │  Reverse Proxy    │
                         │  + Static Files   │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
           ┌────────┴──────┐ ┌───┴──────┐ ┌───┴──────────┐
           │ Streamlit     │ │ FastAPI  │ │ Static Assets │
           │ (:8501)       │ │ (:8000)  │ │ (images/cases)│
           └────────┬──────┘ └───┬──────┘ └──────────────┘
                    │             │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
     ┌────────┴──────┐ ┌───┴──────┐ ┌──┴───────────┐
     │ VLM Server    │ │ Redis    │ │ PostgreSQL    │
     │ (GPU: T4/A10) │ │ (:6379)  │ │ (:5432)       │
     └───────────────┘ └──────────┘ └───────────────┘
```

## Docker Compose

```yaml
# docker-compose.yml
version: "3.8"

services:
  nginx:
    image: nginx:alpine
    ports: ["443:443"]
    volumes: ["./nginx.conf:/etc/nginx/nginx.conf"]
    depends_on: [streamlit, fastapi]

  fastapi:
    build: .
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/emogame
      - REDIS_URL=redis://redis:6379
    depends_on: [postgres, redis]

  streamlit:
    build: .
    command: streamlit run app.py --server.port 8501
    environment:
      - API_BASE_URL=http://fastapi:8000
    depends_on: [fastapi]

  vlm-server:
    build: ./vlm
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_NAME=InternVL2-4B

  redis:
    image: redis:7-alpine

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: emogame
    volumes: ["pgdata:/var/lib/postgresql/data"]

volumes:
  pgdata:
```

## GPU 需求

| 环境 | GPU | VRAM | 可运行模型 | 适用场景 |
|------|-----|------|-----------|----------|
| 最低 | CPU only | — | 传统 CV + API 降级 | 本地开发、demo |
| 开发 | T4 | 16GB | InternVL2-4B (batch=8) | 日常开发调试 |
| 生产 | A10 | 24GB | InternVL2-4B + Qwen2-VL-7B | 正式环境 |

### 无 GPU 降级方案

```python
# vlm/pipeline.py
class VLMPipeline:
    def __init__(self, use_gpu: bool = True):
        if use_gpu and torch.cuda.is_available():
            self.l1_model = InternVL2("InternVL2-4B")
            self.l2_model = Qwen2VL("Qwen2-VL-7B")
        else:
            # 降级: 传统 CV + GPT-4o-mini API
            self.l1_model = TraditionalCV()  # 颜色直方图 + 边缘检测
            self.l2_model = GPTVision()      # GPT-4o-mini vision
            logger.warning("GPU not available, using fallback pipeline")
```

## 开发路线图

```
Phase 1 (Week 1-2): Foundation
├── ✅ 环境搭建: .venv + pip install
├── ✅ 数据加载: hero-skin-image JSON 解析
├── ⬜ VLM 集成: InternVL2-4B 本地部署 + 批量推理脚本
├── ⬜ 爬虫框架: 官方商店 + 贴吧基础爬虫
├── ⬜ 特征存储: SQLite schema + CRUD API
└── ⬜ 基础 Streamlit: 英雄选择 + 皮肤浏览

Phase 2 (Week 3-4): Feature Engineering
├── ⬜ 33 维特征向量完整实现
├── ⬜ VLM 三级管线联调
├── ⬜ 社区数据爬虫 (贴吧 + NGA + Bilibili)
├── ⬜ 特征归一化 + 缺失值处理
└── ⬜ 初始数据集标注 (50 条)

Phase 3 (Week 5-6): Model
├── ⬜ 规则引擎完整实现
├── ⬜ XGBoost 训练 + CV 验证 (目标 MAPE < 25%)
├── ⬜ Ensemble Blending (动态 Alpha)
├── ⬜ 初始权重 AHP 校准
└── ⬜ 模型序列化 + 版本管理

Phase 4 (Week 7-8): Agent & Integration
├── ⬜ LangGraph 智能体工作流
├── ⬜ LLM 报告生成 (GPT-4o-mini)
├── ✅ FastAPI 端点 (/api/skins, /api/evaluate, /api/sales-report)
├── ⬜ Streamlit 完整仪表盘
└── ⬜ 案例库 (10+ 条)

Phase 5 (Week 9-10): Polish & Deploy
├── ⬜ 竞品对比分析模块
├── ⬜ 风险预警完善
├── ⬜ PDF/JSON 导出
├── ⬜ Docker 容器化
├── ⬜ 性能优化 (缓存策略、并发)
└── ⬜ 用户文档 + 演示视频素材
```

## 环境变量

```bash
# .env.example
AUTODL_TOKEN=sk-xxx
DATABASE_URL=sqlite:///data/emogame.db
REDIS_URL=redis://localhost:6379
VLM_MODEL_DIR=/models
BAIDU_INDEX_API_KEY=xxx
BILIBILI_API_KEY=xxx
LOG_LEVEL=INFO
```

## 下一步

- 回到 [文档索引](README.md)
- 开始实施 → Phase 1: Foundation
