"""Streamlit Phase 1 browser for WZRY skin data.

This is the foundation UI from the roadmap: choose a hero, inspect skins, and
seed a baseline feature-store record. Full premium scoring remains Phase 2+.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from crawlers.manager import CrawlerManager, SkinSummary
from feature_engineering.pipeline import FeaturePipeline


st.set_page_config(page_title="EmoGame", layout="wide")


@st.cache_resource
def get_crawler_manager() -> CrawlerManager:
    return CrawlerManager()


@st.cache_resource
def get_feature_pipeline() -> FeaturePipeline:
    return FeaturePipeline(crawler_manager=get_crawler_manager())


def image_path_for(skin: SkinSummary) -> Path | None:
    if not skin.image_path:
        return None
    path = Path(skin.image_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path if path.exists() else None


manager = get_crawler_manager()
features = get_feature_pipeline()

heroes = manager.list_heroes()
weibo_sets = manager.load_weibo_comment_sets()

st.title("EmoGame")
st.caption("Phase 1 foundation browser: WZRY official skin data, local images, and social-ingestion summaries.")

with st.sidebar:
    st.header("Skin Browser")
    if heroes:
        hero_name = st.selectbox("Hero", heroes, index=0)
    else:
        hero_name = ""
        st.warning("No WZRY skin database found under data/wzry_skins/skins.sqlite3.")
    search_text = st.text_input("Search hero or skin")
    show_all = st.checkbox("Show all heroes", value=False)

if search_text.strip():
    skins = manager.search_skins(search_text, limit=100)
elif show_all:
    skins = manager.list_skins(limit=250)
else:
    skins = manager.list_skins(hero_name=hero_name, limit=200)

left, right = st.columns([1, 2])

with left:
    st.subheader("Available Skins")
    if not skins:
        st.info("No skins matched the current filters.")
        selected_skin = None
    else:
        labels = [skin.display_name for skin in skins]
        selected_label = st.selectbox("Skin", labels)
        selected_skin = skins[labels.index(selected_label)]

with right:
    st.subheader("Skin Detail")
    if selected_skin is None:
        st.info("Select a skin to inspect local metadata.")
    else:
        image_path = image_path_for(selected_skin)
        meta_col, image_col = st.columns([1, 1])
        with meta_col:
            st.write(f"**Hero:** {selected_skin.hero_name}")
            st.write(f"**Skin:** {selected_skin.skin_name}")
            st.write(f"**Quality:** {selected_skin.quality or 'unknown'}")
            st.write(f"**Online date:** {selected_skin.online_date or 'unknown'}")
            st.write(f"**Price:** {selected_skin.price_text or 'unknown'}")
            st.write(f"**Source key:** `{selected_skin.source_key}`")
            record = features.get(selected_skin.source_key)
            if record:
                st.success(f"Feature record: {record.source}")
            else:
                st.info("No feature record yet.")
            if st.button("Seed Feature Store", use_container_width=True):
                features.seed_from_official_skin(selected_skin.source_key)
                rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun")
                rerun()
        with image_col:
            if image_path:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)
            else:
                st.warning("No local image path is available for this skin.")

st.divider()
st.subheader("Data Ingestion Status")
metric_cols = st.columns(4)
metric_cols[0].metric("Heroes", len(heroes))
metric_cols[1].metric("Current Skin Rows", len(manager.list_skins(limit=10000)))
metric_cols[2].metric("Weibo Crawl Files", len(weibo_sets))
metric_cols[3].metric("Feature Records", len(features.store.list(limit=10000)))

if weibo_sets:
    st.dataframe(weibo_sets, use_container_width=True, hide_index=True)
else:
    st.info("No Weibo comment JSON files found under data/weibo_comments/.")
