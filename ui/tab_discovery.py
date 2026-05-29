import html
import streamlit as st
from ui.components import render_card_grid, section_header
from storage.db import log_recommendations


def _to_display(items: list[dict]) -> list[dict]:
    return [
        {
            "title":       m.get("title", ""),
            "year":        str(m.get("year", "")),
            "rating":      m.get("rating", 0),
            "tmdb_rating": m.get("rating", 0),
            "poster_url":  m.get("poster_url", ""),
            "badge": (m.get("language", "").upper()
                      if m.get("language") not in ("en", None, "") else None),
        }
        for m in items
    ]


def render(content_engine, models: dict, session_id: str):
    st.markdown("### Discover movies similar to one you loved")

    # ── Inputs ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        seed_input = st.text_input(
            "Movie title",
            placeholder='e.g. "Baahubali", "Inception", "Ra One"',
            key="disc_seed",
        )
    with col2:
        mood = st.selectbox(
            "Mood", ["Any", "Intense", "Light", "Mind-bending", "Emotional"],
            key="disc_mood",
        )
    with col3:
        language = st.selectbox(
            "Language", ["All", "Indian only", "Hollywood only"],
            key="disc_lang",
        )

    search_clicked = st.button("🔍 Search", type="primary", key="disc_search")

    if not seed_input:
        st.info("Enter a movie title and click Search.")
        return

    # only run search when button is explicitly clicked
    cache_key = f"disc_{seed_input.strip().lower()}_{mood}_{language}"
    if search_clicked or cache_key in st.session_state:
        if search_clicked or cache_key not in st.session_state:
            with st.spinner(f'Searching for "{seed_input}"...'):
                result = content_engine.recommend(
                    seed_input,
                    mood=mood if mood != "Any" else None,
                    language_filter=language if language != "All" else None,
                )
            st.session_state[cache_key] = result
            # log only on new search
            if "shelves" in result:
                all_titles = [m["title"] for items in result["shelves"].values()
                              for m in items]
                log_recommendations(
                    session_id, tab="discovery", method="content",
                    recommendations=[(t, 0.0) for t in all_titles],
                    seed_movie=result.get("seed", {}).get("title"),
                )
    else:
        return

    result = st.session_state.get(cache_key)
    if not result:
        return

    if "error" in result and "seed" not in result:
        st.error(result["error"])
        return

    seed    = result.get("seed", {})
    shelves = result.get("shelves", {})

    # ── Seed movie card ─────────────────────────────────────────────────
    s_title    = html.escape(str(seed.get("title", "")))
    s_year     = html.escape(str(seed.get("year", "")))
    s_genres   = html.escape(" · ".join(seed.get("genres", [])[:3]))
    s_director = html.escape(str(seed["director"])) if seed.get("director") else ""
    dir_html   = f" · Dir: {s_director}" if s_director else ""
    lang_badge = (f' <span style="background:#333;color:#aaa;font-size:10px;'
                  f'padding:2px 6px;border-radius:4px;">'
                  f'{html.escape(str(seed.get("language", "")).upper())}</span>'
                  if seed.get("language") not in ("en", None, "") else "")
    st.markdown(f"""
    <div style="background:#1a1a2e;border-radius:10px;padding:16px;margin-bottom:20px;
                display:flex;gap:16px;border-left:4px solid #E50914;align-items:flex-start;">
      <img src="{seed.get('poster_url','')}" style="width:80px;height:120px;
           object-fit:cover;border-radius:6px;flex-shrink:0;">
      <div>
        <p style="color:#E50914;font-size:11px;margin:0;letter-spacing:1px;">SEED MOVIE</p>
        <p style="color:white;font-size:18px;font-weight:bold;margin:4px 0;">
          {s_title} ({s_year}){lang_badge}</p>
        <p style="color:#f5c518;margin:2px 0;">★ {seed.get('rating',0):.1f}</p>
        <p style="color:#aaa;font-size:12px;margin:2px 0;">
          {s_genres}{dir_html}</p>
      </div>
    </div>""", unsafe_allow_html=True)

    if not shelves:
        st.warning(result.get("error", "No similar movies found for these filters."))
        return

    for shelf_name, items in shelves.items():
        section_header(f"🎬 {shelf_name}")
        render_card_grid(_to_display(items), cols=4)
