# EmoGame

> 游戏虚拟商品情绪溢价评估智能体 (Game Virtual Goods Emotional Premium Evaluation Agent)

B2B SaaS 智能体，量化游戏虚拟商品（皮肤、特效、表情、头像框等）的情绪溢价。
输入商品属性 → 输出：情绪溢价指数 (0-100)、五维雷达图、建议定价区间、业务风险预警、竞品对比。

## 技术栈

- **语言**: Python 3.12
- **后端**: FastAPI
- **前端**: Streamlit
- **VLM**: InternVL2-4B + Qwen2-VL-7B + GPT-4o-mini
- **ML**: XGBoost + scikit-learn + Optuna
- **Agent**: LangGraph + LangChain
- **数据**: SQLite → PostgreSQL, Redis

## 快速开始

```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动 Streamlit 前端
streamlit run app.py

# 启动 FastAPI 后端
uvicorn api.main:app --reload --port 8000
```

## 项目结构

```
emogame/
├── app.py                    # Streamlit 主入口
├── api/                      # FastAPI 后端
├── agents/                   # LangGraph 智能体
├── vlm/                      # VLM 视觉管线
├── crawlers/                 # 数据爬虫
├── feature_engineering/      # 特征工程
├── models/                   # 情绪溢价模型
├── business/                 # 商业分析
├── data/                     # 数据存储
├── notebooks/                # Jupyter 实验
├── tests/                    # 测试
├── docs/                     # 技术文档
├── hero-skin-image/          # Git submodule (皮肤数据)
├── requirements.txt
└── Dockerfile
```

## 文档

详细技术方案请参阅 [docs/](docs/) 目录：

| 编号 | 文档 | 内容 |
|------|------|------|
| — | [文档索引](docs/README.md) | 完整文档导航 |
| 01 | [总体架构](docs/01-architecture.md) | 四层架构、技术选型 |
| 02 | [VLM 视觉管线](docs/02-vlm-pipeline.md) | 三级模型、Prompt 工程 |
| 03 | [数据采集](docs/03-data-crawling.md) | 爬虫架构、反爬策略 |
| 04 | [特征工程](docs/04-feature-engineering.md) | 33维特征、五维度映射 |
| 05 | [模型层](docs/05-model-layer.md) | 规则引擎 + XGBoost |
| 06 | [智能体架构](docs/06-agent-architecture.md) | LangGraph 多 Agent |
| 07 | [商业分析](docs/07-business-analysis.md) | 定价/风险/竞品 |
| 08 | [用户工作流](docs/08-user-workflow.md) | Streamlit 前端 |
| 09 | [部署运维](docs/09-deployment.md) | Docker、GPU 需求 |

## Python 环境

Python 3.12 虚拟环境位于 `.venv/`。激活：

```bash
source .venv/bin/activate
```

## 当前状态

- ✅ Git submodule `hero-skin-image` 已接入（130英雄、893皮肤、898张图片）
- ✅ 皮肤数据查询 API 测试通过
- ⬜ 技术方案已完成，等待实施
