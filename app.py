import streamlit as st
import pickle

from engines.collaborative import FunkSVDRecommender
from engines.content_based import ContentEngine
from storage.db import init_db, new_session_id
from config import MODELS_DIR

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  body, .stApp { background-color: #0F0F0F; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    color: #aaa; background: #1a1a1a; border-radius: 6px; padding: 6px 20px;
  }
  .stTabs [aria-selected="true"] {
    background: #E50914 !important; color: white !important;
  }
  .stSelectbox label, .stRadio label, .stTextInput label { color: #ccc !important; }
  div[data-testid="stMetricValue"] { color: #E50914; }
  .stButton > button { background: #E50914; color: white; border: none; border-radius: 6px; }
  .stButton > button:hover { background: #c00; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_models():
    with open(f"{MODELS_DIR}/meta.pkl", "rb") as f:
        meta = pickle.load(f)
    funk_svd = FunkSVDRecommender.load(f"{MODELS_DIR}/funk_svd.pkl")
    content  = ContentEngine()
    return meta, funk_svd, content


init_db()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = new_session_id()

meta, funk_svd, content_engine = load_models()

models = {"funk_svd": funk_svd, "content": content_engine}

st.markdown("""
<div style="text-align:center;padding:20px 0 8px 0;">
  <h1 style="color:white;font-size:2.2rem;margin:0;">🎬 Movie Recommender</h1>
  <p style="color:#aaa;margin:4px 0 0 0;">
    Funk SVD · Content Embeddings · MovieLens 1M + TMDB
  </p>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🎯  Watch History",
    "🔍  Discover by Movie",
    "📊  Analytics & Evaluation",
])

session_id = st.session_state["session_id"]

with tab1:
    from ui.tab_watchhistory import render as render_wh
    render_wh(models, meta, session_id)

with tab2:
    from ui.tab_discovery import render as render_disc
    render_disc(content_engine, models, session_id)

with tab3:
    from ui.tab_analytics import render as render_analytics
    render_analytics()
