import html
import streamlit as st
from config import TMDB_IMAGE_PLACEHOLDER


def movie_card(title: str, year, rating, poster_url: str,
               predicted_score: float = None, badge: str = None) -> str:
    title = html.escape(str(title))
    score_html = (
        f'<p style="color:#aaa;font-size:11px;margin:4px 0 0 0;">'
        f'Predicted: &#11088; {predicted_score:.2f}</p>'
        if predicted_score is not None else ""
    )
    badge_html = (
        f'<span style="background:#E50914;color:white;font-size:10px;'
        f'padding:2px 6px;border-radius:4px;font-weight:bold;">{html.escape(str(badge))}</span><br>'
        if badge else ""
    )
    poster    = poster_url or TMDB_IMAGE_PLACEHOLDER
    year_int  = None
    if year is not None:
        try:
            year_int = int(str(year))
        except (ValueError, TypeError):
            pass
    year_str   = f" ({year_int})" if year_int else ""
    rating_val = rating if isinstance(rating, (int, float)) else 0.0
    rating_str = f"&#9733; {rating_val:.1f}" if rating_val else ""

    return (
        f'<div style="background:#1a1a2e;border-radius:10px;overflow:hidden;'
        f'box-shadow:0 4px 12px rgba(0,0,0,0.4);">'
        f'<img src="{poster}" style="width:100%;height:260px;object-fit:cover;" '
        f'onerror="this.src=\'{TMDB_IMAGE_PLACEHOLDER}\'">'
        f'<div style="padding:10px;">'
        f'{badge_html}'
        f'<p style="color:white;font-weight:bold;margin:4px 0 2px 0;'
        f'font-size:13px;line-height:1.3;">{title}{year_str}</p>'
        f'<p style="color:#f5c518;margin:0;font-size:12px;">{rating_str}</p>'
        f'{score_html}'
        f'</div></div>'
    )


def render_card_grid(movies: list[dict], cols: int = 5):
    """Render all cards in one st.markdown call using CSS grid — avoids
    the Streamlit columns + unsafe_allow_html rendering bug."""
    if not movies:
        return
    cards = "".join(
        movie_card(
            title=m.get("title", ""),
            year=m.get("year") or m.get("release_year"),
            rating=m.get("tmdb_rating") or m.get("rating") or 0,
            poster_url=m.get("poster_url", ""),
            predicted_score=m.get("predicted_score"),
            badge=m.get("badge"),
        )
        for m in movies
    )
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);'
        f'gap:12px;margin-bottom:20px;">{cards}</div>',
        unsafe_allow_html=True,
    )


def taste_profile_card(genres: list, era: str, avg_rating: float, total_rated: int):
    genres_str = " &middot; ".join(f"<b>{g}</b>" for g in genres[:3]) if genres else "&#8212;"
    st.markdown(
        f'<div style="background:#1a1a2e;border-radius:10px;padding:16px;'
        f'margin-bottom:16px;border-left:4px solid #E50914;">'
        f'<p style="color:#aaa;font-size:12px;margin:0 0 8px 0;letter-spacing:1px;">TASTE PROFILE</p>'
        f'<p style="color:white;margin:4px 0;">Top genres: {genres_str}</p>'
        f'<p style="color:white;margin:4px 0;">Era preference: <b>{era}</b></p>'
        f'<p style="color:white;margin:4px 0;">Avg rating given: <b>&#11088; {avg_rating:.1f}</b></p>'
        f'<p style="color:#aaa;margin:4px 0;font-size:12px;">{total_rated} movies rated in history</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(text: str):
    st.markdown(
        f'<h3 style="color:white;border-bottom:1px solid #333;'
        f'padding-bottom:6px;margin-top:24px;">{text}</h3>',
        unsafe_allow_html=True,
    )
