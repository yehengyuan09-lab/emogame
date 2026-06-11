# EmoGame 技术文档

> 游戏虚拟商品情绪溢价评估智能体 — 完整技术解决方案

## 文档索引

| 编号 | 文档 | 内容 |
|------|------|------|
| 00 | [Python 环境与本地脚本入门](00-python-env-and-scripts.md) | Python 环境、依赖安装、数据采集脚本运行说明 |
| 01 | [总体架构](01-architecture.md) | 四层架构设计、技术选型总览、系统分层职责 |
| 02 | [VLM 视觉语言模型](02-vlm-pipeline.md) | 三级管线设计、模型选型、Prompt 工程、批量推理优化 |
| 03 | [数据采集系统](03-data-crawling.md) | 多源数据矩阵、爬虫架构、反爬策略 |
| 04 | [特征工程系统](04-feature-engineering.md) | 31 维特征定义、五维度映射、归一化策略 |
| 05 | [情绪溢价模型层](05-model-layer.md) | 规则引擎、XGBoost 回归、Ensemble 融合 |
| 06 | [智能体执行架构](06-agent-architecture.md) | LangGraph 工作流、多 Agent 协作、LLM 报告生成 |
| 07 | [商业价值分析](07-business-analysis.md) | 定价引擎、风险预警、竞品对比、案例库 |
| 08 | [用户工作流与前端](08-user-workflow.md) | 端到端用户旅程、Streamlit 页面设计 |
| 09 | [部署与运维](09-deployment.md) | Docker 部署、GPU 需求、开发路线图 |

## 快速导航

### 按角色

- **后端开发者** → [01 架构](01-architecture.md) → [04 特征工程](04-feature-engineering.md) → [05 模型层](05-model-layer.md)
- **算法/ML 工程师** → [02 VLM](02-vlm-pipeline.md) → [04 特征工程](04-feature-engineering.md) → [05 模型层](05-model-layer.md)
- **前端开发者** → [01 架构](01-architecture.md) → [08 用户工作流](08-user-workflow.md)
- **DevOps** → [09 部署](09-deployment.md)
- **产品/项目经理** → [01 架构](01-architecture.md) → [07 商业分析](07-business-analysis.md) → [08 用户工作流](08-user-workflow.md)

### 按开发阶段

| 阶段 | 对应文档 | 核心交付 |
|------|----------|----------|
| 基础搭建 | 01, 09 | 环境搭建、项目骨架、依赖安装 |
| 数据准备 | 03, 04 | 爬虫运行、数据集标注、特征存储 |
| 模型开发 | 02, 05 | VLM 部署、规则引擎、XGBoost 训练 |
| 智能体集成 | 06 | LangGraph 工作流、LLM 报告生成 |
| 产品化 | 07, 08 | Streamlit 仪表盘、定价/风险/竞品模块 |

## 项目目录结构

```
emogame/
├── app.py                          # Streamlit 主入口
├── api/                            # FastAPI 后端
│   ├── main.py
│   └── routes/
│       ├── evaluation.py
│       ├── cases.py
│       └── export.py
├── agents/                         # 智能体层
│   ├── workflow.py                 # LangGraph 工作流
│   ├── tools.py                    # Agent 工具定义
│   └── prompts.py                  # LLM Prompt 模板
├── vlm/                            # VLM 视觉管线
│   ├── pipeline.py                 # 三级管线编排
│   ├── internvl.py                 # InternVL2 封装
│   ├── qwen_vl.py                  # Qwen2-VL 封装
│   └── preprocess.py               # 图像预处理
├── crawlers/                       # 数据采集
│   ├── manager.py                  # 爬虫调度器
│   ├── official_store.py           # 官方商城爬虫
│   ├── community.py                # 社区爬虫
│   └── market.py                   # 二级市场爬虫
├── feature_engineering/            # 特征工程
│   ├── pipeline.py                 # 特征主管线
│   ├── features.py                 # 特征向量定义
│   └── normalizers.py              # 归一化策略
├── models/                         # 模型层
│   ├── rule_engine.py              # 规则引擎
│   ├── xgboost_model.py            # XGBoost 模型
│   ├── ensemble.py                 # 模型融合
│   └── train.py                    # 训练脚本
├── business/                       # 商业分析
│   ├── pricing.py                  # 定价引擎
│   ├── risk.py                     # 风险分析
│   ├── competitor.py               # 竞品分析
│   └── case_library.py             # 案例库
├── data/                           # 数据层
│   ├── feature_store.py            # 特征存储
│   ├── schemas.py                  # 数据库 Schema
│   └── migrations/
├── notebooks/                      # Jupyter 实验
│   ├── 01_data_exploration.ipynb
│   ├── 02_vlm_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_validation.ipynb
├── tests/                          # 测试
├── docs/                           # 技术文档
├── hero-skin-image/                # Git submodule (皮肤图片数据)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

*文档版本: v1.0 | 最后更新: 2026-06-05*
