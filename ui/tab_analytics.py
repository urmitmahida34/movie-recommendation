import streamlit as st
import plotly.express as px
import pandas as pd
from storage.db import get_analytics


def render():
    st.markdown("### Analytics & Model Info")

    # ── Funk SVD model card ────────────────────────────────────────────────
    st.markdown("#### About the Recommendation Model")
    st.markdown("""
    <div style="background:#1a1a2e;border-radius:10px;padding:16px;
                margin-bottom:20px;border-left:4px solid #E50914;">
      <p style="color:#E50914;font-size:11px;margin:0 0 8px 0;letter-spacing:1px;">MODEL</p>
      <p style="color:white;font-size:16px;font-weight:bold;margin:0 0 6px 0;">
        Funk SVD — Regularised Matrix Factorisation</p>
      <p style="color:#aaa;font-size:12px;margin:0 0 12px 0;">
        Trained on MovieLens 1M · 80/20 chronological split · 497 sampled users evaluated
      </p>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
        <div style="background:#111;border-radius:8px;padding:12px;text-align:center;">
          <p style="color:#E50914;font-size:22px;font-weight:bold;margin:0;">0.054</p>
          <p style="color:#aaa;font-size:11px;margin:4px 0 0 0;">Precision@10</p>
        </div>
        <div style="background:#111;border-radius:8px;padding:12px;text-align:center;">
          <p style="color:#E50914;font-size:22px;font-weight:bold;margin:0;">0.060</p>
          <p style="color:#aaa;font-size:11px;margin:4px 0 0 0;">NDCG@10</p>
        </div>
        <div style="background:#111;border-radius:8px;padding:12px;text-align:center;">
          <p style="color:#E50914;font-size:22px;font-weight:bold;margin:0;">0.888</p>
          <p style="color:#aaa;font-size:11px;margin:4px 0 0 0;">RMSE</p>
        </div>
      </div>
      <p style="color:#555;font-size:11px;margin:12px 0 0 0;">
        n_factors=100 · n_epochs=20 · lr=0.005 · reg=0.02 (SGD + L2 regularisation)
      </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Live session analytics ─────────────────────────────────────────────
    st.markdown("#### Session Analytics")
    analytics = get_analytics()
    total = analytics["total"]

    if total == 0:
        st.info("No recommendations logged yet. Use the app to generate recommendations.")
        return

    st.metric("Total Recommendations Served", total)

    col3, col4 = st.columns(2)

    top_movies = analytics["top_movies"]
    if top_movies:
        with col3:
            df_top = pd.DataFrame(top_movies)
            fig4 = px.bar(df_top, x="cnt", y="recommended_title",
                          orientation="h",
                          title="Top 10 Most Recommended",
                          color_discrete_sequence=["#E50914"],
                          template="plotly_dark")
            fig4.update_layout(paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig4, use_container_width=True)

    seed_movies = analytics["seed_movies"]
    if seed_movies:
        with col4:
            df_seed = pd.DataFrame(seed_movies)
            fig_seed = px.bar(df_seed, x="cnt", y="seed_movie",
                              orientation="h",
                              title="Most Searched Seed Movies",
                              color_discrete_sequence=["#888"],
                              template="plotly_dark")
            fig_seed.update_layout(paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
                                   yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_seed, use_container_width=True)

    daily = analytics["daily"]
    if daily:
        df_d = pd.DataFrame(daily)
        max_cnt = df_d["cnt"].max()
        y_upper = max_cnt + max(2, int(max_cnt * 0.2))
        fig5 = px.line(df_d, x="day", y="cnt",
                       title="Daily Recommendation Volume",
                       markers=True,
                       color_discrete_sequence=["#E50914"],
                       template="plotly_dark")
        fig5.update_layout(
            paper_bgcolor="#0F0F0F",
            plot_bgcolor="#0F0F0F",
            yaxis=dict(
                range=[0, y_upper],
                tick0=0,
                dtick=max(1, max_cnt // 5),
                title="Recommendations",
            ),
            xaxis_title="Date",
        )
        fig5.update_traces(mode="lines+markers", marker=dict(size=8))
        st.plotly_chart(fig5, use_container_width=True)

    if not top_movies and not seed_movies and not daily:
        st.markdown("**Most Searched Seed Movies**")
        st.dataframe(pd.DataFrame(seed_movies), use_container_width=True, hide_index=True)
