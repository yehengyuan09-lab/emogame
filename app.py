"""Streamlit workbench for local skin evaluation and sales actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from business.sales_advisor import SalesAdvisor
from data.market_signal_repository import MarketSignalRepository
from data.skin_repository import DEFAULT_DB_PATH, SkinRepository
from feature_engineering.features import MarketValidationSignals
from feature_engineering.pipeline import FeatureBuilder
from models.rule_engine import RuleEngine


ASPECT_LABELS = {
    "visual_appeal": "观感",
    "in_game_feel": "手感",
    "craftsmanship_quality": "品质",
    "collection_value": "收藏",
    "value_for_money": "性价比",
    "purchase_intent": "购买意愿",
    "market_heat": "市场热度",
}


def page_config() -> None:
    st.set_page_config(
        page_title="EmoGame Workbench",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
        [data-testid="stMetric"] { border: 1px solid #e6e8ef; padding: 10px 12px; border-radius: 6px; }
        [data-testid="stSidebar"] { border-right: 1px solid #e6e8ef; }
        .small-muted { color: #667085; font-size: 0.86rem; }
        .decision { font-weight: 700; font-size: 1.05rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def search_skins(db_path: str, query: str, limit: int) -> list[dict[str, Any]]:
    repo = SkinRepository(Path(db_path))
    if not Path(db_path).exists():
        return []
    return repo.search_skins(query, limit=limit) if query else repo.list_skins(limit=limit)


def build_payload(
    db_path: Path,
    source_key: str,
    signals: MarketValidationSignals,
) -> dict[str, Any]:
    repo = SkinRepository(db_path)
    features = FeatureBuilder(repo).build(source_key, signals)
    evaluation = RuleEngine().evaluate(features)
    report = SalesAdvisor().advise(features, evaluation)
    return {
        "skin": features.to_dict(),
        "evaluation": evaluation.to_dict(),
        "sales_report": report.to_dict(),
    }


def load_signals(db_path: Path, source_key: str, mode: str) -> MarketValidationSignals:
    if mode == "忽略证据":
        return MarketValidationSignals()
    if mode == "手动模拟":
        return MarketValidationSignals(
            visual_score=st.session_state.get("visual_score", 70) / 100,
            feel_score=st.session_state.get("feel_score", 70) / 100,
            craftsmanship_score=st.session_state.get("craftsmanship_score", 70) / 100,
            collection_score=st.session_state.get("collection_score", 70) / 100,
            value_score=st.session_state.get("value_score", 70) / 100,
            purchase_intent_score=st.session_state.get("purchase_intent_score", 70) / 100,
            sentiment_score=st.session_state.get("sentiment_score", 70) / 100,
            discussion_count=st.session_state.get("discussion_count", 1000),
            video_views=st.session_state.get("video_views", 100000),
            marketing_volume=st.session_state.get("marketing_volume", 5000),
            sales_volume=st.session_state.get("sales_volume", None) or None,
            ownership_rate=st.session_state.get("ownership_rate", 0) / 100 or None,
        )
    return MarketSignalRepository(db_path).get_signals(source_key)


def render_sidebar() -> tuple[Path, str | None, MarketValidationSignals | None]:
    st.sidebar.title("EmoGame")
    st.sidebar.caption("本地皮肤评估与销售动作工作台")

    db_text = st.sidebar.text_input("SQLite 数据库", value=str(DEFAULT_DB_PATH))
    db_path = Path(db_text)
    if not db_path.exists():
        st.sidebar.error("数据库不存在。先运行采集脚本生成 skins.sqlite3。")
        return db_path, None, None

    query = st.sidebar.text_input("搜索皮肤", value="龙胆")
    rows = search_skins(str(db_path), query, 40)
    if not rows:
        st.sidebar.warning("没有匹配皮肤。")
        return db_path, None, None

    labels = [f"{row['source_key']} | {row['hero_name']} / {row['skin_name']}" for row in rows]
    selected_label = st.sidebar.selectbox("选择皮肤", labels)
    selected_index = labels.index(selected_label)
    source_key = str(rows[selected_index]["source_key"])

    mode = st.sidebar.segmented_control(
        "证据信号",
        ["数据库证据", "忽略证据", "手动模拟"],
        default="数据库证据",
    )
    if mode == "手动模拟":
        with st.sidebar.expander("模拟信号", expanded=True):
            st.slider("观感", 0, 100, 78, key="visual_score")
            st.slider("手感", 0, 100, 72, key="feel_score")
            st.slider("品质", 0, 100, 76, key="craftsmanship_score")
            st.slider("收藏", 0, 100, 70, key="collection_score")
            st.slider("性价比", 0, 100, 62, key="value_score")
            st.slider("购买意愿", 0, 100, 74, key="purchase_intent_score")
            st.slider("整体情绪", 0, 100, 78, key="sentiment_score")
            st.number_input("讨论量", min_value=0, value=8000, step=500, key="discussion_count")
            st.number_input("B站播放量", min_value=0, value=1000000, step=50000, key="video_views")
            st.number_input("传播互动量", min_value=0, value=30000, step=1000, key="marketing_volume")
            st.number_input("销量", min_value=0, value=0, step=1000, key="sales_volume")
            st.slider("拥有率 (%)", 0, 100, 0, key="ownership_rate")

    signals = load_signals(db_path, source_key, mode)
    return db_path, source_key, signals


def render_header(payload: dict[str, Any]) -> None:
    report = payload["sales_report"]
    evaluation = payload["evaluation"]
    st.title(f"{report['hero_name']} / {report['skin_name']}")
    st.caption("证据优先评估。分数不是销量预测，销售动作必须由市场证据验证。")

    cols = st.columns(5)
    cols[0].metric("销售决策", report["decision"])
    cols[1].metric("销售准备度", f"{report['sales_readiness']}/100")
    score = evaluation["evaluation_score"] if evaluation["evaluation_score"] is not None else "N/A"
    cols[2].metric("证据分", score)
    cols[3].metric("官方先验", f"{evaluation['official_prior_score']}/100")
    cols[4].metric("置信度", f"{evaluation['confidence']:.2f}")

    st.markdown(
        f"<p class='decision'>{report['decision_reason']}</p>",
        unsafe_allow_html=True,
    )


def render_sales_tab(payload: dict[str, Any]) -> None:
    report = payload["sales_report"]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("购买驱动力")
        for item in report["purchase_drivers"]:
            st.info(item)
    with right:
        st.subheader("转化阻力")
        blockers = report["conversion_blockers"] or ["暂无明确阻力。"]
        for item in blockers:
            st.warning(item)

    st.subheader("建议动作")
    for item in report["recommended_actions"]:
        st.write(f"**{item['action']}**")
        st.write(item["detail"])

    pricing = report["pricing_guidance"]
    st.subheader("价格动作")
    st.write(f"**{pricing['posture']}**")
    st.write(pricing["rationale"])
    if pricing.get("official_price_text"):
        st.caption(f"官方价格文本：{pricing['official_price_text']}")


def render_evidence_tab(payload: dict[str, Any]) -> None:
    evaluation = payload["evaluation"]
    report = payload["sales_report"]
    aspect_scores = evaluation["aspect_scores"]
    chart_data = {
        "维度": [ASPECT_LABELS.get(name, name) for name in aspect_scores],
        "分数": [score or 0 for score in aspect_scores.values()],
    }
    st.bar_chart(chart_data, x="维度", y="分数", height=260)

    cols = st.columns(2)
    with cols[0]:
        st.subheader("证据状态")
        st.write(f"验证状态：`{evaluation['validation_status']}`")
        st.write(f"证据覆盖：`{evaluation['evidence_coverage']}`")
        st.write("市场字段：")
        st.write(evaluation["evidence"]["market_signal_fields"] or "无")
    with cols[1]:
        st.subheader("缺失证据")
        for gap in report["evidence_gaps"] or ["none"]:
            st.write(f"- `{gap}`")


def render_skin_tab(payload: dict[str, Any]) -> None:
    skin = payload["skin"]
    fields = {
        "source_key": skin["source_key"],
        "hero_id": skin["hero_id"],
        "skin_id": skin["skin_id"],
        "quality": skin["quality"],
        "online_date": skin["online_date"],
        "acquire_method": skin["acquire_method"],
        "price_text": skin["price_text"],
        "official_tier": skin["official_tier"],
        "hero_skin_count": skin["hero_skin_count"],
    }
    st.dataframe(
        [{"字段": key, "值": value} for key, value in fields.items()],
        hide_index=True,
        use_container_width=True,
    )


def main() -> None:
    page_config()
    db_path, source_key, signals = render_sidebar()
    if not source_key or signals is None:
        st.title("EmoGame Workbench")
        st.write("请选择一个皮肤开始评估。")
        return

    try:
        payload = build_payload(db_path, source_key, signals)
    except ValueError as exc:
        st.error(str(exc))
        return

    render_header(payload)
    tabs = st.tabs(["销售动作", "证据结构", "皮肤信息", "JSON"])
    with tabs[0]:
        render_sales_tab(payload)
    with tabs[1]:
        render_evidence_tab(payload)
    with tabs[2]:
        render_skin_tab(payload)
    with tabs[3]:
        st.download_button(
            "下载 JSON",
            data=json.dumps(payload, ensure_ascii=False, indent=2),
            file_name=f"{source_key}-sales-report.json",
            mime="application/json",
        )
        st.json(payload)


if __name__ == "__main__":
    main()
