# 06 — 智能体执行架构

## 多智能体协作系统

基于 **LangGraph** 构建有状态的多智能体工作流，6 个 Agent 协作完成端到端评估。

```
                        ┌──────────────────────┐
                        │   Orchestrator Agent  │
                        │   (主管智能体)          │
                        │   - 任务分解与调度       │
                        │   - 状态管理            │
                        │   - 异常处理            │
                        └──────────┬───────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
    ┌──────┴──────┐         ┌──────┴──────┐         ┌──────┴──────┐
    │ Collector   │         │  Analyzer   │         │  Reporter   │
    │ Agent       │         │  Agent      │         │  Agent      │
    │             │         │             │         │             │
    │ - 数据采集   │         │ - VLM 触发  │         │ - LLM 报告  │
    │ - 爬虫调度   │         │ - 特征工程  │         │ - 策略建议  │
    │ - 数据清洗   │         │ - 模型推理  │         │ - 可视化数据│
    └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                        ┌──────────┴───────────┐
                        │   Shared State        │
                        │   (Pydantic Models)    │
                        └──────────────────────┘
```

## Agent 职责

| Agent | 触发条件 | 输入 | 输出 | 超时 |
|-------|----------|------|------|------|
| **Collector** | 用户提交评估请求 | skin_id, hero_name | raw_data (多源字典) | 30s |
| **VLM Analyzer** | Collector 完成 | raw_data.image_paths | vlm_features | 10s/img |
| **Feature Engineer** | VLM Analyzer 完成 | vlm_features + market + community | feature_vector (33 维) | 5s |
| **Model Inferer** | Feature Engineer 完成 | feature_vector | premium_result | 2s |
| **Business Analyzer** | Model Inferer 完成 | premium + features | pricing + risks + competitors | 10s |
| **Report Generator** | Business Analyzer 完成 | 全部上游结果 | LLM 自然语言报告 | 15s |

## LangGraph 工作流定义

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    # 输入
    skin_id: str
    hero_name: str
    game_genre: str          # MOBA / FPS / RPG / Gacha

    # 中间产物
    raw_data: dict           # Collector 输出
    vlm_features: dict       # VLM Analyzer 输出
    feature_vector: dict     # Feature Engineer 输出
    premium_result: dict     # Model Inferer 输出
    business_analysis: dict  # Business Analyzer 输出
    report: str              # Report Generator 输出

    # 控制
    errors: Annotated[list, operator.add]
    current_stage: str


def create_evaluation_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("collect_data", collector_node)
    workflow.add_node("vlm_analyze", vlm_node)
    workflow.add_node("feature_engineer", feature_engineer_node)
    workflow.add_node("model_inference", model_inference_node)
    workflow.add_node("business_analyze", business_analysis_node)
    workflow.add_node("generate_report", report_generation_node)
    workflow.add_node("error_handler", error_handler_node)

    # 定义边
    workflow.set_entry_point("collect_data")
    workflow.add_edge("collect_data", "vlm_analyze")
    workflow.add_edge("vlm_analyze", "feature_engineer")
    workflow.add_edge("feature_engineer", "model_inference")
    workflow.add_edge("model_inference", "business_analyze")
    workflow.add_edge("business_analyze", "generate_report")
    workflow.add_edge("generate_report", END)

    # 条件边
    workflow.add_conditional_edges(
        "vlm_analyze",
        lambda s: "error_handler" if s["errors"] else "feature_engineer",
    )

    return workflow.compile()
```

## Agent 节点实现

```python
async def collector_node(state: AgentState) -> AgentState:
    crawler = get_crawler_manager()
    state["raw_data"] = await crawler.collect_all(
        skin_id=state["skin_id"], hero_name=state["hero_name"]
    )
    state["current_stage"] = "data_collected"
    return state

async def vlm_node(state: AgentState) -> AgentState:
    try:
        vlm = get_vlm_pipeline()
        state["vlm_features"] = await vlm.analyze(
            state["raw_data"]["image_paths"]["wallpaper_big"]
        )
    except Exception as e:
        state["errors"].append(f"VLM error: {e}")
    state["current_stage"] = "vlm_done"
    return state

async def feature_engineer_node(state: AgentState) -> AgentState:
    fe = get_feature_pipeline()
    vector = await fe.build_vector(
        vlm_features=state["vlm_features"],
        market_data=state["raw_data"]["market"],
        community_data=state["raw_data"]["community"],
    )
    state["feature_vector"] = vector.to_dict()
    state["current_stage"] = "features_ready"
    return state

async def model_inference_node(state: AgentState) -> AgentState:
    predictor = get_ensemble_predictor()
    state["premium_result"] = predictor.predict(state["feature_vector"])
    state["current_stage"] = "inference_done"
    return state

async def business_analysis_node(state: AgentState) -> AgentState:
    analyzer = get_business_analyzer()
    state["business_analysis"] = await analyzer.analyze(
        premium_result=state["premium_result"],
        feature_vector=state["feature_vector"],
        hero_name=state["hero_name"],
    )
    return state

async def report_generation_node(state: AgentState) -> AgentState:
    llm = get_llm_client()
    state["report"] = await llm.generate_report(
        premium=state["premium_result"],
        business=state["business_analysis"],
        skin_id=state["skin_id"],
    )
    state["current_stage"] = "completed"
    return state
```

## LLM 报告生成 System Prompt

```
你是游戏商业化分析师，专精于MOBA游戏虚拟物品定价策略。

报告结构：
1. **情绪溢价总览** (2-3句总结)
2. **五维度雷达解读** (每个维度1-2句分析)
3. **定价区间建议** (给出具体人民币区间及理由)
4. **风险提示** (列出2-3条业务风险)
5. **竞品参照** (与同类皮肤对比)
6. **运营策略建议** (2-3条可执行建议)

要求：数据驱动，语气专业但不失亲和，每条建议可落地执行，使用中文输出。
```

## Agent Tools 定义

```python
from langchain.tools import tool

@tool
def search_hero_skins(hero_name: str) -> list[dict]:
    """查询指定英雄的所有皮肤列表"""

@tool
def query_skin_price(skin_name: str) -> dict:
    """查询指定皮肤的历史定价和当前价格"""

@tool
def get_vlm_analysis(skin_id: str) -> dict:
    """触发 VLM 对皮肤图片进行视觉分析"""

@tool
def fetch_community_sentiment(hero_name: str, days: int = 30) -> dict:
    """获取社区情感数据（近N天讨论量、正面/负面比例）"""

@tool
def compare_similar_skins(skin_id: str, top_k: int = 5) -> list[dict]:
    """查找同类皮肤并返回对比数据"""

@tool
def calculate_premium(features: dict) -> dict:
    """输入特征向量，返回情绪溢价指数和五维度分"""

@tool
def get_market_trends(game_genre: str) -> dict:
    """获取特定游戏品类的市场趋势数据"""
```

## 错误处理与重试

```
                    ┌──────────┐
                    │  Node    │
                    │  Execute │
                    └────┬─────┘
                         │
                    ┌────▼─────┐     yes     ┌──────────┐
                    │  Error?  │────────────►│  Retry   │
                    └────┬─────┘             │  (max 3) │
                         │ no                └────┬─────┘
                    ┌────▼─────┐                  │
                    │  Next    │◄─────────────────┘
                    │  Node    │  (success)
                    └──────────┘
                         │
                    (3 failures)
                         │
                    ┌────▼─────┐
                    │  Error   │
                    │  Handler │──► 记录日志 + 降级 + 通知用户
                    └──────────┘
```

## 下一步

- → [07 — 商业价值分析](07-business-analysis.md)
