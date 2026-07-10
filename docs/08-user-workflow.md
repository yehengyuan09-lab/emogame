# 08 — 用户工作流与前端

## 端到端用户旅程

```
  [用户]                          [系统]                         [输出]
    │                               │                              │
    │  ① 输入皮肤信息                │                              │
    │  - 选择游戏 (MOBA/FPS/RPG)     │                              │
    │  - 输入英雄名 / 皮肤名          │                              │
    │  - 上传皮肤图片 (可选)           │                              │
    │──────────────────────────────►│                              │
    │                               │                              │
    │                               │  ② Collector Agent            │
    │                               │  - 查询官方数据               │
    │                               │  - 爬取社区讨论 + 百度指数    │
    │                               │  - 获取二级市场行情           │
    │                               │                              │
    │                               │  ③ VLM Pipeline              │
    │                               │  qwen2.5vl L1/L2             │
    │                               │  → GPT5.4-mini               │
    │                               │                              │
    │                               │  ④ Feature Engineering       │
    │                               │  33 维特征向量               │
    │                               │                              │
    │                               │  ⑤ Model Inference           │
    │                               │  Rule Engine + XGBoost       │
    │                               │  → Ensemble Blending         │
    │                               │                              │
    │                               │  ⑥ Business Analysis         │
    │                               │  定价建议 + 风险 + 竞品       │
    │                               │                              │
    │                               │  ⑦ LLM Report Generation     │
    │                               │                              │
    │  ⑧ 返回结构化报告              │                              │
    │◄──────────────────────────────│                              │
    │                               │                              │
    │  ┌─────────────────────────┐  │                              │
    │  │ 📊 情绪溢价指数: 85/100  │  │                              │
    │  │                         │  │                              │
    │  │ [五维度雷达图]           │  │   雷达图 (Plotly)            │
    │  │   审美  ████░░ 70       │  │                              │
    │  │   归属  █████░ 92       │  │                              │
    │  │   炫耀  ████░  80       │  │                              │
    │  │   收集  █████  90       │  │                              │
    │  │   惊喜  ███░░  60       │  │                              │
    │  │                         │  │                              │
    │  │ 💰 建议定价: ¥88-178    │  │                              │
    │  │ ⚠️ 风险: 返场过频        │  │                              │
    │  │ 📋 竞品: 李白-凤求凰     │  │                              │
    │  │ 📝 策略建议: ...         │  │                              │
    │  └─────────────────────────┘  │                              │
    │                               │                              │
    │  ⑨ 交互操作                    │                              │
    │  - What-if 参数调整            │                              │
    │  - 导出 PDF / JSON 报告        │                              │
    │  - 保存到案例库                │                              │
    │                               │                              │
```

## Streamlit 前端页面结构

```
┌──────────────────────────────────────────────────────────────┐
│  🎮 EmoGame 情绪溢价评估                                       │
├───────────────┬──────────────────────────────────────────────┤
│  📋 皮肤录入   │                                              │
│               │  ┌─────────────────────────────────────┐     │
│  游戏         │  │  📊 情绪溢价指数: 85/100             │     │
│  [王者荣耀  ▾]│  │                                     │     │
│               │  │  ┌─────────────────────────────────┐│     │
│  英雄名       │  │  │         五维度雷达图              ││     │
│  [孙悟空    ] │  │  │      (Plotly Radar Chart)        ││     │
│               │  │  └─────────────────────────────────┘│     │
│  皮肤名       │  │                                     │     │
│  [至尊宝    ] │  │  📝 策略报告                         │     │
│               │  │  ┌─────────────────────────────────┐│     │
│  ──────────── │  │  │ 该皮肤情绪溢价指数为85分，属于... ││     │
│  高级选项     │  │  │ (LLM 生成的自然语言报告)          ││     │
│  [展开 ▸]    │  │  └─────────────────────────────────┘│     │
│               │  └─────────────────────────────────────┘     │
│  [🔍 开始评估]│                                              │
│               │  ┌──────────┬──────────┬──────────────────┐  │
│               │  │ 💰 定价   │ ⚠️ 风险   │ 📋 竞品对比       │  │
│               │  │ ¥88-178  │ 返场过频  │ 李白-凤求凰: 82   │  │
│               │  └──────────┴──────────┴──────────────────┘  │
├───────────────┴──────────────────────────────────────────────┤
│  [📄 导出 PDF]  [📊 导出 JSON]  [💾 保存到案例库]             │
└──────────────────────────────────────────────────────────────┘
```

### 核心代码结构

```python
# app.py
import streamlit as st
from agents.workflow import create_evaluation_graph

st.set_page_config(page_title="EmoGame", page_icon="🎮", layout="wide")

# 侧边栏：输入
with st.sidebar:
    st.title("📋 皮肤信息录入")
    game = st.selectbox("游戏", ["王者荣耀", "英雄联盟手游", "原神", "崩坏：星穹铁道"])
    hero_name = st.text_input("英雄/角色名")
    skin_name = st.text_input("皮肤名称")
    with st.expander("高级特征输入"):
        official_tier = st.selectbox("官方稀有度", ["勇者", "史诗", "传说", "无双", "荣耀典藏"])
        is_limited = st.checkbox("限定皮肤")
        official_price = st.number_input("官方定价 (¥)", min_value=0.0)
    uploaded_image = st.file_uploader("上传皮肤图片 (可选)", type=["jpg", "png", "webp"])
    submitted = st.button("🔍 开始评估", type="primary", use_container_width=True)

# 主区域
st.title("🎮 EmoGame 情绪溢价评估")

if submitted:
    with st.spinner("正在评估..."):
        workflow = create_evaluation_graph()
        result = workflow.invoke({
            "skin_id": f"{hero_name}-{skin_name}",
            "hero_name": hero_name,
            "game_genre": "MOBA",
        })

    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric("情绪溢价指数", f"{result['premium_result']['total_premium']}/100")
        st.subheader("五维度雷达图")
        st.plotly_chart(plot_radar(result["premium_result"]["sub_scores"]))
        st.subheader("📝 策略报告")
        st.markdown(result.get("report", ""))

    with col2:
        st.subheader("💰 定价建议")
        p = result["business_analysis"]["pricing"]
        st.metric("建议价格区间", f"¥{p['min']} - ¥{p['max']}")
        st.subheader("⚠️ 风险预警")
        for risk in result["business_analysis"]["risks"]:
            st.warning(f"**{risk['type']}**: {risk['message']}")
        st.subheader("📋 竞品对比")
        for comp in result["business_analysis"]["competitors"]:
            st.info(f"**{comp['name']}**: 溢价 {comp['premium']}/100")

    # 导出
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.download_button("📄 导出 PDF", data=generate_pdf(result), file_name="report.pdf")
    with col_e2:
        st.download_button("📊 导出 JSON", data=generate_json(result), file_name="data.json")
    with col_e3:
        if st.button("💾 保存到案例库"):
            save_to_case_library(result)
            st.success("已保存!")
```

## 交互时序（预估延迟）

| 步骤 | 操作 | 预估延迟 | 用户可见 |
|------|------|----------|----------|
| 1 | 用户输入 + 提交 | 即时 | ✅ spinner "正在采集数据..." |
| 2 | Collector Agent | 5-15s | spinner |
| 3 | VLM Pipeline | 3-6s/img | spinner (可显示 "分析皮肤图片中...") |
| 4 | Feature Engineering | 1-2s | spinner |
| 5 | Model Inference | <1s | spinner |
| 6 | Business Analysis | 2-5s | spinner |
| 7 | LLM Report | 5-10s | spinner "生成策略报告中..." |
| 8 | 渲染结果 | 即时 | ✅ 全量报告展示 |
| **总计** | | **20-40s** | |

## 下一步

- → [09 — 部署与运维](09-deployment.md)

## 当前落地入口

本地 Streamlit 工作台已经可运行：

```bash
python -m streamlit run app.py --server.port 8501
```

当前版本聚焦运营验收，不依赖 VLM 或 LLM 在线推理：

- 左侧搜索并选择本地 SQLite 中的皮肤。
- 支持读取数据库证据、忽略证据、手动模拟证据信号。
- 主区展示销售决策、销售准备度、证据分、官方先验和置信度。
- Tab 展示销售动作、证据结构、皮肤信息和 JSON 导出。
