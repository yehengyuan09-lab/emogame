"""Streamlit workbench for transparent skin evaluation and sales calibration."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st

from business.sales_advisor import SalesAdvisor
from data.market_signal_repository import MarketSignalRepository
from data.skin_repository import DEFAULT_DB_PATH, SkinRepository
from feature_engineering.features import MarketValidationSignals
from feature_engineering.pipeline import FeatureBuilder
from models.rule_engine import RuleEngine
from models.sales_calibration import RbfSalesCalibrator, calibration_features
from models.sales_deviation import compare_score_to_sales, sales_blind_signals


CALIBRATION_MODEL_PATH = Path("outputs/sales_calibration_model.json")
CALIBRATION_REPORT_PATH = Path("outputs/sales_calibration_report.json")

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
        page_title="EmoGame Evidence Workbench",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
        [data-testid="stMetric"] {
            border: 1px solid #e6e8ef;
            padding: 10px 12px;
            border-radius: 6px;
            background: #ffffff;
        }
        [data-testid="stSidebar"] { border-right: 1px solid #e6e8ef; }
        .small-muted { color: #667085; font-size: 0.86rem; }
        .audit-note {
            border-left: 4px solid #4b5563;
            background: #f8fafc;
            padding: 0.75rem 0.9rem;
            margin: 0.5rem 0 1rem;
        }
        .risk-note {
            border-left: 4px solid #b45309;
            background: #fffbeb;
            padding: 0.75rem 0.9rem;
            margin: 0.5rem 0 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def search_skins(db_path: str, query: str, limit: int) -> list[dict[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        return []
    repo = SkinRepository(path)
    return repo.search_skins(query, limit=limit) if query else repo.list_skins(limit=limit)


@st.cache_data(show_spinner=False)
def dataset_summary(db_path: str) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return empty_summary()

    skin_repo = SkinRepository(path)
    market_repo = MarketSignalRepository(path)
    market_repo.ensure_schema()
    stats = skin_repo.stats()
    evidence_rows = all_evidence_rows(path)
    basis_counts = Counter(classify_sales_basis(row["metrics"]) for row in evidence_rows)
    platform_counts = Counter(row["platform"] for row in evidence_rows)
    sales_source_keys = market_repo.list_source_keys_with_sales_evidence()

    return {
        "game": "王者荣耀",
        "implemented_games": 1,
        "candidate_games": 0,
        "skins": stats["skins"],
        "heroes": stats["heroes"],
        "assets": stats["assets"],
        "skins_with_detail": stats["with_detail"],
        "skins_missing_detail": stats["missing_detail"],
        "sales_evidence_skins": len(sales_source_keys),
        "sales_evidence_items": len(evidence_rows),
        "basis_counts": dict(sorted(basis_counts.items())),
        "platform_counts": dict(sorted(platform_counts.items())),
        "source_keys": sales_source_keys,
    }


def empty_summary() -> dict[str, Any]:
    return {
        "game": "王者荣耀",
        "implemented_games": 1,
        "candidate_games": 0,
        "skins": 0,
        "heroes": 0,
        "assets": 0,
        "skins_with_detail": 0,
        "skins_missing_detail": 0,
        "sales_evidence_skins": 0,
        "sales_evidence_items": 0,
        "basis_counts": {},
        "platform_counts": {},
        "source_keys": [],
    }


def all_evidence_rows(db_path: Path, *, source_key: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
    query = """
        SELECT source_key, platform, external_id, url, title, author,
               published_at, metrics_json, collected_at
        FROM opinion_evidence_items
    """
    params: list[Any] = []
    if source_key:
        query += " WHERE source_key = ?"
        params.append(source_key)
    query += " ORDER BY collected_at DESC, evidence_id DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(query, params).fetchall()]

    for row in rows:
        row["metrics"] = json.loads(row.pop("metrics_json") or "{}")
        row["basis"] = classify_sales_basis(row["metrics"])
        row["confidence"] = row["metrics"].get("source_confidence")
        row["metric_summary"] = metric_summary(row["metrics"])
    return rows


def classify_sales_basis(metrics: dict[str, Any]) -> str:
    if any(key in metrics for key in ("sales_volume", "units_sold", "sales")):
        return "exact_volume"
    if any(key in metrics for key in ("estimated_sales_volume", "sales_volume_estimate")):
        return "estimated_volume"
    if any(key in metrics for key in ("sales_rank", "hot_sales_rank", "rank")):
        return "rank_proxy"
    if any(key in metrics for key in ("sales_volume_upper_bound", "sales_upper_bound")):
        return "upper_bound"
    if any(key in metrics for key in ("sales_volume_lower_bound", "sales_lower_bound")):
        return "lower_bound"
    return "non_sales_signal"


def metric_summary(metrics: dict[str, Any]) -> str:
    keys = [
        "sales_volume",
        "estimated_sales_volume",
        "sales_volume_upper_bound",
        "sales_rank",
        "rank_size",
        "source_confidence",
    ]
    parts = [f"{key}={metrics[key]}" for key in keys if key in metrics]
    return ", ".join(parts) if parts else json.dumps(metrics, ensure_ascii=False)


def build_payload(
    db_path: Path,
    source_key: str,
    signals: MarketValidationSignals,
    calibration_model: RbfSalesCalibrator | None = None,
) -> dict[str, Any]:
    repo = SkinRepository(db_path)
    market_repo = MarketSignalRepository(db_path)
    features = FeatureBuilder(repo).build(source_key, signals)
    evaluation = RuleEngine().evaluate(features)
    report = SalesAdvisor().advise(features, evaluation)
    evidence = market_repo.list_evidence(source_key)

    score_features = FeatureBuilder(repo).build(source_key, sales_blind_signals(signals))
    gap_evaluation = RuleEngine().evaluate(score_features)
    sales_gap = compare_score_to_sales(features, gap_evaluation, evidence)
    if calibration_model and sales_gap["sales_score"] is not None:
        sales_gap = apply_calibration(score_features, gap_evaluation, sales_gap, calibration_model)

    return {
        "skin": features.to_dict(),
        "evaluation": evaluation.to_dict(),
        "score_evaluation": gap_evaluation.to_dict(),
        "sales_report": report.to_dict(),
        "sales_gap": sales_gap,
        "evidence_items": evidence,
    }


def apply_calibration(
    score_features: Any,
    gap_evaluation: Any,
    sales_gap: dict[str, Any],
    calibration_model: RbfSalesCalibrator,
) -> dict[str, Any]:
    calibrated_score = calibration_model.predict(calibration_features(score_features, gap_evaluation))
    calibrated_gap = calibrated_score - int(sales_gap["sales_score"])
    updated = dict(sales_gap)
    updated["base_score"] = sales_gap["score"]
    updated["base_gap"] = sales_gap["gap"]
    updated["calibrated_score"] = calibrated_score
    updated["calibrated_gap"] = calibrated_gap
    updated["score"] = calibrated_score
    updated["score_basis"] = "calibrated_sales_score"
    updated["gap"] = calibrated_gap
    updated["absolute_gap"] = abs(calibrated_gap)
    updated["gap_direction"] = (
        "aligned"
        if abs(calibrated_gap) <= 8
        else "score_above_sales"
        if calibrated_gap > 0
        else "sales_above_score"
    )
    updated["warnings"] = list(updated["warnings"]) + ["calibrated_on_small_public_sample"]
    return updated


def load_calibration_model(path: Path = CALIBRATION_MODEL_PATH) -> RbfSalesCalibrator | None:
    if not path.exists():
        return None
    return RbfSalesCalibrator.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))


def load_calibration_report(path: Path = CALIBRATION_REPORT_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def render_sidebar() -> tuple[Path, str | None, MarketValidationSignals | None, RbfSalesCalibrator | None]:
    st.sidebar.title("EmoGame")
    st.sidebar.caption("证据审计、评分、销量偏差和校准工作台")

    st.sidebar.selectbox("游戏", ["王者荣耀（已接入）"], index=0)
    st.sidebar.caption("其他游戏还没有接入，不会伪装成可计算。")

    db_text = st.sidebar.text_input("SQLite 数据库", value=str(DEFAULT_DB_PATH))
    db_path = Path(db_text)
    if not db_path.exists():
        st.sidebar.error("数据库不存在。先运行采集脚本生成 skins.sqlite3。")
        return db_path, None, None, None

    query = st.sidebar.text_input("搜索皮肤", value="龙胆")
    rows = search_skins(str(db_path), query, 60)
    if not rows:
        st.sidebar.warning("没有匹配的皮肤。")
        return db_path, None, None, None

    labels = [f"{row['source_key']} | {row['hero_name']} / {row['skin_name']}" for row in rows]
    selected_label = st.sidebar.selectbox("选择皮肤", labels)
    source_key = str(rows[labels.index(selected_label)]["source_key"])

    mode = st.sidebar.segmented_control(
        "评分证据",
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
            st.number_input("视频播放量", min_value=0, value=1000000, step=50000, key="video_views")
            st.number_input("传播互动量", min_value=0, value=30000, step=1000, key="marketing_volume")
            st.number_input("销量", min_value=0, value=0, step=1000, key="sales_volume")
            st.slider("拥有率 (%)", 0, 100, 0, key="ownership_rate")

    use_calibration = st.sidebar.toggle("加载 ML 校准模型", value=CALIBRATION_MODEL_PATH.exists())
    calibration_model = load_calibration_model() if use_calibration else None
    if use_calibration and calibration_model is None:
        st.sidebar.warning("未找到 outputs/sales_calibration_model.json。")

    signals = load_signals(db_path, source_key, mode)
    return db_path, source_key, signals, calibration_model


def render_header(payload: dict[str, Any], summary: dict[str, Any]) -> None:
    skin = payload["skin"]
    gap = payload["sales_gap"]
    evaluation = payload["evaluation"]
    report = payload["sales_report"]

    st.title(f"{skin['hero_name']} / {skin['skin_name']}")
    st.caption("当前只接入王者荣耀。销量结果来自公开证据代理，不是官方全量销量库。")

    cols = st.columns(6)
    cols[0].metric("皮肤库", f"{summary['skins']}")
    cols[1].metric("销量证据皮肤", f"{summary['sales_evidence_skins']}")
    cols[2].metric("原始评分", score_text(evaluation))
    cols[3].metric("当前评分", gap["score"] if gap["score"] is not None else "N/A")
    cols[4].metric("销量分", gap["sales_score"] if gap["sales_score"] is not None else "N/A")
    cols[5].metric("Gap", gap["gap"] if gap["gap"] is not None else "N/A")

    if "calibrated_score" in gap:
        st.markdown(
            "<div class='risk-note'>当前显示的是 ML 校准分。原始评分仍保留；校准模型只在 32 条公开样例上训练，"
            "不能当成已泛化的销量预测。</div>",
            unsafe_allow_html=True,
        )
    elif gap["sales_score"] is None:
        st.markdown(
            "<div class='risk-note'>当前皮肤没有可用销量证据，只能看评分和缺口，不应做销售结论。</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='audit-note'>{report['decision_reason']}</div>",
            unsafe_allow_html=True,
        )


def score_text(evaluation: dict[str, Any]) -> str:
    score = evaluation["evaluation_score"]
    return str(score) if score is not None else f"{evaluation['official_prior_score']} 先验"


def render_method_tab(calibration_report: dict[str, Any] | None) -> None:
    st.subheader("系统工作原理")
    st.markdown(
        """
| 步骤 | 输入 | 输出 | 审计点 |
|---|---|---|---|
| 1. 皮肤基础库 | 官方皮肤、英雄、品质、上架时间、获取方式 | `SkinFeatureVector` | 只说明皮肤是什么，不代表销量 |
| 2. 舆情/市场证据 | B站、微博、人工导入维度分、讨论量 | `evaluation_score` | 缺证据时只给官方先验 |
| 3. 销量证据 | 公开销量榜、估算销量、上限/下限声明 | `sales_score` | 必须标明证据类型和链接 |
| 4. 偏差检验 | `score - sales_score` | `gap` 和方向 | 判断模型低估/高估销量 |
| 5. ML 校准 | sales-blind 特征，不含销量字段 | `calibrated_score` | 只作校准层，不覆盖原始评分 |
        """
    )
    st.markdown(
        "<div class='risk-note'>关键限制：现在没有官方全量销量 API。公开榜单是代理证据，样本量仍很小，"
        "所以界面必须把来源、置信度和训练集/验证集结果展示出来。</div>",
        unsafe_allow_html=True,
    )

    if calibration_report:
        base = calibration_report["baseline"]
        calibrated = calibration_report["calibrated"]
        loo = calibration_report["leave_one_out"]
        cols = st.columns(4)
        cols[0].metric("校准样本", calibrated["n"])
        cols[1].metric("训练集超差", f"{calibrated['exceedances']}/{calibrated['n']}")
        cols[2].metric("二项检验 p", calibrated["binomial_p_value_at_target_probability"])
        cols[3].metric("留一超差", f"{loo['exceedances']}/{loo['n']}" if loo.get("available") else "N/A")
        st.write(
            f"Baseline：{base['exceedances']}/{base['n']} 个样本 |gap| > 10，"
            f"MAE={base['mae']}。校准后训练集通过，但 leave-one-out 未通过，说明还需要更多样本。"
        )
    else:
        st.info("未加载校准报告。运行 scripts/calibrate_sales_score.py 可生成 outputs/sales_calibration_report.json。")


def render_dataset_tab(db_path: Path, summary: dict[str, Any]) -> None:
    st.subheader("数据审计")
    cols = st.columns(5)
    cols[0].metric("游戏数", summary["implemented_games"])
    cols[1].metric("英雄", summary["heroes"])
    cols[2].metric("皮肤", summary["skins"])
    cols[3].metric("有销量证据皮肤", summary["sales_evidence_skins"])
    cols[4].metric("销量证据条目", summary["sales_evidence_items"])

    st.markdown(
        "<div class='audit-note'>这不是生产级样本量。当前销量校准样本是公开证据样例；"
        "要服务销量，需要持续扩充到每个游戏数百条、每条有可追溯来源。</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.write("证据类型")
        st.dataframe(
            [{"basis": key, "count": value} for key, value in summary["basis_counts"].items()],
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.write("平台来源")
        st.dataframe(
            [{"platform": key, "count": value} for key, value in summary["platform_counts"].items()],
            hide_index=True,
            use_container_width=True,
        )

    st.write("公开销量证据明细")
    st.dataframe(
        evidence_table(all_evidence_rows(db_path, limit=300)),
        hide_index=True,
        use_container_width=True,
    )


def render_gap_tab(payload: dict[str, Any]) -> None:
    gap = payload["sales_gap"]
    cols = st.columns(5)
    cols[0].metric("Base score", gap.get("base_score", gap["score"]))
    cols[1].metric("Displayed score", gap["score"] if gap["score"] is not None else "N/A")
    cols[2].metric("Sales score", gap["sales_score"] if gap["sales_score"] is not None else "N/A")
    cols[3].metric("Gap", gap["gap"] if gap["gap"] is not None else "N/A")
    cols[4].metric("Evidence confidence", f"{gap['confidence']:.2f}")

    st.write(f"方向：`{gap['gap_direction']}`")
    st.write(gap["interpretation"])
    if gap.get("sales_evidence"):
        st.write("当前采用的销量证据")
        st.dataframe(evidence_table([flatten_gap_evidence(gap)]), hide_index=True, use_container_width=True)
    st.write("警告")
    for warning in gap["warnings"] or ["none"]:
        st.write(f"- `{warning}`")


def flatten_gap_evidence(gap: dict[str, Any]) -> dict[str, Any]:
    evidence = gap["sales_evidence"]
    return {
        "source_key": gap["source_key"],
        "platform": "sales_public",
        "title": evidence.get("source_title"),
        "url": evidence.get("source_url"),
        "basis": evidence.get("basis"),
        "confidence": evidence.get("confidence"),
        "metric_summary": metric_summary(
            {
                "estimated_sales_volume": evidence.get("volume"),
                "sales_rank": evidence.get("rank"),
                "rank_size": evidence.get("rank_size"),
                "source_confidence": evidence.get("confidence"),
            }
        ),
    }


def render_current_sources_tab(payload: dict[str, Any]) -> None:
    st.subheader("当前皮肤证据")
    items = payload["evidence_items"]
    if not items:
        st.warning("当前皮肤没有证据条目。")
        return
    rows = []
    for item in items:
        metrics = item.get("metrics") or {}
        rows.append(
            {
                "source_key": item["source_key"],
                "platform": item["platform"],
                "title": item.get("title"),
                "url": item.get("url"),
                "basis": classify_sales_basis(metrics),
                "confidence": metrics.get("source_confidence"),
                "metric_summary": metric_summary(metrics),
            }
        )
    st.dataframe(evidence_table(rows), hide_index=True, use_container_width=True)


def evidence_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_key": row.get("source_key"),
            "platform": row.get("platform"),
            "basis": row.get("basis"),
            "confidence": row.get("confidence"),
            "title": row.get("title"),
            "url": row.get("url"),
            "metrics": row.get("metric_summary"),
        }
        for row in rows
    ]


def render_sales_tab(payload: dict[str, Any]) -> None:
    report = payload["sales_report"]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("购买驱动力")
        for item in report["purchase_drivers"]:
            st.info(item)
    with right:
        st.subheader("转化阻力")
        for item in report["conversion_blockers"] or ["暂无明确阻力。"]:
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


def render_evaluation_tab(payload: dict[str, Any]) -> None:
    evaluation = payload["evaluation"]
    report = payload["sales_report"]
    aspect_scores = evaluation["aspect_scores"]
    if any(score is not None for score in aspect_scores.values()):
        chart_data = {
            "维度": [ASPECT_LABELS.get(name, name) for name in aspect_scores],
            "分数": [score or 0 for score in aspect_scores.values()],
        }
        st.bar_chart(chart_data, x="维度", y="分数", height=260)
    else:
        st.info("当前皮肤没有可审计的维度评分，系统只会显示官方先验和缺失证据。")

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


def render_game_scope_tab(summary: dict[str, Any]) -> None:
    st.subheader("游戏扩展状态")
    st.markdown(
        "<div class='risk-note'>当前没有做跨游戏通用评分。不同游戏的价格体系、稀缺机制、抽卡/直售、"
        "二级市场和舆情空间都不同，必须先接入各自的数据适配器。</div>",
        unsafe_allow_html=True,
    )
    rows = [
        {
            "game_scope": "王者荣耀",
            "status": "已接入本地皮肤库",
            "current_data": f"{summary['skins']} skins / {summary['sales_evidence_skins']} with sales evidence",
            "next_requirement": "补真实销量、拥有率、舆情维度评分",
        },
        {
            "game_scope": "新 MOBA / 射击 / 抽卡游戏",
            "status": "未接入",
            "current_data": "0 production samples",
            "next_requirement": "实现游戏元数据 crawler、价格/获取方式解析、销量代理证据 importer",
        },
        {
            "game_scope": "跨游戏对比",
            "status": "未开放",
            "current_data": "不能直接比较",
            "next_requirement": "先做游戏内归一化，再做跨游戏层级校准",
        },
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_json_tab(payload: dict[str, Any], source_key: str) -> None:
    st.download_button(
        "下载 JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name=f"{source_key}-sales-audit.json",
        mime="application/json",
    )
    st.json(payload)


def main() -> None:
    page_config()
    db_path, source_key, signals, calibration_model = render_sidebar()
    if not source_key or signals is None:
        st.title("EmoGame Evidence Workbench")
        st.write("请选择一个皮肤开始审计。")
        return

    try:
        payload = build_payload(db_path, source_key, signals, calibration_model)
        summary = dataset_summary(str(db_path))
        calibration_report = load_calibration_report()
    except ValueError as exc:
        st.error(str(exc))
        return

    render_header(payload, summary)
    tabs = st.tabs(["工作原理", "数据审计", "评分/销量偏差", "当前证据", "销售动作", "评分结构", "游戏扩展", "JSON"])
    with tabs[0]:
        render_method_tab(calibration_report)
    with tabs[1]:
        render_dataset_tab(db_path, summary)
    with tabs[2]:
        render_gap_tab(payload)
    with tabs[3]:
        render_current_sources_tab(payload)
    with tabs[4]:
        render_sales_tab(payload)
    with tabs[5]:
        render_evaluation_tab(payload)
    with tabs[6]:
        render_game_scope_tab(summary)
    with tabs[7]:
        render_json_tab(payload, source_key)


if __name__ == "__main__":
    main()
