import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import data.tmdb_client as tmdb
from config import (EMBEDDING_MODEL, TMDB_GENRE_IDS, TMDB_GENRE_NAMES,
                    INDIAN_LANGUAGE_CODES, MOOD_KEYWORD_MAP)

_embed_model = None

# shelf display names keyed by candidate source
SHELF_LABELS = {
    "similar":  "More Like This",
    "reco":     "Recommended For You",
    "discover": "Same Genre",
}


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _build_text(movie: dict) -> str:
    parts = []
    if movie.get("title"):
        parts.append(movie["title"])
    overview = (movie.get("overview") or "")[:300]
    if overview:
        parts.append(overview)
    if movie.get("genres"):
        parts.append("Genres: " + ", ".join(movie["genres"]))
    if movie.get("keywords"):
        parts.append("Keywords: " + ", ".join(movie["keywords"][:15]))
    if movie.get("director"):
        parts.append("Director: " + movie["director"])
    if movie.get("cast"):
        parts.append("Cast: " + ", ".join(movie["cast"][:3]))
    return ". ".join(parts)


def _from_search_result(m: dict, source: str) -> dict:
    """Build a lightweight candidate from a TMDB search/list result —
    no extra API calls. genre_ids are mapped to names locally."""
    genres = [TMDB_GENRE_NAMES[g] for g in m.get("genre_ids", [])
              if g in TMDB_GENRE_NAMES]
    return {
        "tmdb_id":   m.get("id"),
        "title":     m.get("title", ""),
        "year":      (m.get("release_date") or "")[:4],
        "rating":    m.get("vote_average", 0),
        "popularity": m.get("popularity", 0),
        "overview":  m.get("overview", ""),
        "genres":    genres,
        "language":  m.get("original_language", "en"),
        "poster_url": tmdb.get_poster_url(m.get("poster_path")),
        "source":    source,
    }


def _apply_filters(candidates: list[dict], mood: str,
                   language_filter: str) -> list[dict]:
    if language_filter == "Indian only":
        candidates = [c for c in candidates
                      if c.get("language") in INDIAN_LANGUAGE_CODES]
    elif language_filter == "Hollywood only":
        candidates = [c for c in candidates if c.get("language") == "en"]

    if mood and mood != "Any":
        mood_words = set(MOOD_KEYWORD_MAP.get(mood, []))
        def _match(c):
            genres = set(g.lower() for g in c.get("genres", []))
            return bool(mood_words & genres)
        filtered = [c for c in candidates if _match(c)]
        # don't wipe out all results if mood is too narrow — fall back
        candidates = filtered or candidates
    return candidates


class ContentEngine:
    # NOTE: the embedding model is intentionally NOT loaded here. It loads
    # lazily on the first recommend() call (see _get_embed_model). This keeps
    # torch/sentence-transformers out of memory at startup so the app boots
    # under tight memory limits (e.g. Streamlit Community Cloud ~1GB). Cost:
    # the first Discover search pays a one-time ~10s model-load.
    def recommend(self, seed_title: str, mood: str = None,
                  language_filter: str = None) -> dict:
        # ── 1. Find + enrich seed (only the seed needs full enrichment) ──
        raw = tmdb.search_movie(seed_title)
        if not raw:
            return {"error": f"Could not find '{seed_title}' on TMDB."}

        seed_id = raw["id"]
        seed    = tmdb.enrich_movie(seed_id)

        # ── 2. Gather candidates from 3 sources (no per-movie enrich) ────
        similar_raw = tmdb.get_similar_movies(seed_id)
        reco_raw    = tmdb.get_recommendations(seed_id)

        genre_ids = [TMDB_GENRE_IDS[g] for g in seed.get("genres", [])
                     if g in TMDB_GENRE_IDS][:2]
        discover_raw = []
        if genre_ids:
            discover_raw = tmdb.get_discover_movies(
                genre_ids, language=seed.get("language")
            )[:20]
            if seed.get("language") != "en":
                discover_raw += tmdb.get_discover_movies(genre_ids)[:10]

        # build candidates straight from list results, dedup, keep source
        seen_ids: set[int] = {seed_id}
        candidates: list[dict] = []
        for m, source in ([(x, "similar")  for x in similar_raw] +
                          [(x, "reco")     for x in reco_raw] +
                          [(x, "discover") for x in discover_raw]):
            mid = m.get("id")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                candidates.append(_from_search_result(m, source))

        if not candidates:
            return {"seed": seed, "shelves": {},
                    "error": "No candidates found."}

        # ── 3. Mood / language filters (use fields already present) ──────
        candidates = _apply_filters(candidates, mood, language_filter)
        if not candidates:
            return {"seed": seed, "shelves": {},
                    "error": "No results after applying filters."}

        # ── 4. Embed seed + candidates, rank by cosine similarity ────────
        model  = _get_embed_model()
        texts  = [_build_text(seed)] + [_build_text(c) for c in candidates]
        embeds = model.encode(texts, convert_to_numpy=True,
                              show_progress_bar=False, batch_size=64)

        sims = cosine_similarity(embeds[0:1], embeds[1:])[0]
        for i, c in enumerate(candidates):
            c["_sim"] = float(sims[i])
        candidates.sort(key=lambda x: x["_sim"], reverse=True)

        # ── 5. Group into shelves by source (max 4 each), best-sim first ─
        shelves: dict[str, list] = {}
        for c in candidates:
            label = SHELF_LABELS.get(c["source"], "More Like This")
            shelves.setdefault(label, [])
            if len(shelves[label]) < 4:
                shelves[label].append(c)

        # drop empty shelves, preserve a sensible order
        ordered = {}
        for label in ["More Like This", "Recommended For You", "Same Genre"]:
            if shelves.get(label):
                ordered[label] = shelves[label]

        return {"seed": seed, "shelves": ordered}
