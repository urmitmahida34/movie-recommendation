import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import TMDB_API_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE_URL, TMDB_IMAGE_PLACEHOLDER, TMDB_CACHE_DIR

os.makedirs(TMDB_CACHE_DIR, exist_ok=True)

# persistent session: reuses one TLS connection across all calls (HTTP
# keep-alive) instead of a fresh handshake per request — major speedup
# when hitting the same host many times.
_session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=20,
    pool_maxsize=20,
    max_retries=Retry(total=2, backoff_factor=0.3,
                      status_forcelist=[502, 503, 504]),
)
_session.mount("https://", _adapter)


def _cache_path(key: str) -> str:
    safe = key.replace("/", "_").replace("?", "_").replace("&", "_")
    return os.path.join(TMDB_CACHE_DIR, f"{safe}.json")


def _get(endpoint: str, params: dict = None) -> dict | None:
    cache_file = _cache_path(endpoint + str(sorted((params or {}).items())))
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    try:
        resp = _session.get(
            f"{TMDB_BASE_URL}{endpoint}",
            params={"api_key": TMDB_API_KEY, **(params or {})},
            timeout=(3.05, 6),  # (connect, read) — fail fast on slow reads
        )
    except requests.exceptions.RequestException:
        return None
    if resp.status_code != 200:
        return None

    data = resp.json()
    with open(cache_file, "w") as f:
        json.dump(data, f)
    return data


def search_movie(title: str, year: int = None) -> dict | None:
    from rapidfuzz import fuzz as rfuzz
    # year is NOT sent as a TMDB filter — MovieLens/TMDB years often differ
    # by one, and a strict filter then returns zero results (no poster).
    # It's used as a soft ranking boost below instead.
    data = _get("/search/movie", {"query": title, "include_adult": False})
    if not data or not data.get("results"):
        return None
    results = data["results"]

    title_lower = title.lower().strip()

    def _rank(r):
        t     = (r.get("title") or "").lower()
        ot    = (r.get("original_title") or "").lower()
        exact = (t == title_lower or ot == title_lower)
        sim = 1.0 if exact else max(
            rfuzz.ratio(title_lower, t),
            rfuzz.ratio(title_lower, ot),
            rfuzz.partial_ratio(title_lower, t) * 0.9,
        ) / 100.0
        pop = min(r.get("popularity", 0) / 200.0, 1.0)
        # soft year proximity: full boost at exact year, fading over ±3 yrs
        year_boost = 0.0
        if year:
            ry = (r.get("release_date") or "")[:4]
            if ry.isdigit():
                year_boost = max(0.0, 1.0 - abs(int(ry) - year) / 3.0)
        score = sim * 0.70 + pop * 0.10 + year_boost * 0.10
        if exact:
            score += 0.50  # exact title always ranks above fuzzy matches
        return score

    results.sort(key=_rank, reverse=True)
    return results[0]


def get_movie_details(movie_id: int) -> dict | None:
    return _get(f"/movie/{movie_id}")


def get_movie_keywords(movie_id: int) -> list[str]:
    data = _get(f"/movie/{movie_id}/keywords")
    if not data:
        return []
    return [kw["name"] for kw in data.get("keywords", [])]


def get_movie_credits(movie_id: int) -> dict:
    data = _get(f"/movie/{movie_id}/credits")
    if not data:
        return {"director": None, "cast": []}
    director = next(
        (p["name"] for p in data.get("crew", []) if p.get("job") == "Director"), None
    )
    cast = [p["name"] for p in data.get("cast", [])[:5]]
    return {"director": director, "cast": cast}


def get_similar_movies(movie_id: int) -> list[dict]:
    data = _get(f"/movie/{movie_id}/similar")
    return data.get("results", []) if data else []


def get_recommendations(movie_id: int) -> list[dict]:
    data = _get(f"/movie/{movie_id}/recommendations")
    return data.get("results", []) if data else []


def get_poster_url(poster_path: str | None) -> str:
    if not poster_path:
        return TMDB_IMAGE_PLACEHOLDER
    return f"{TMDB_IMAGE_BASE_URL}{poster_path}"


def get_discover_movies(genre_ids: list[int], language: str = None,
                        page: int = 1) -> list[dict]:
    params = {
        "with_genres": ",".join(str(g) for g in genre_ids),
        "sort_by": "popularity.desc",
        "page": page,
    }
    if language:
        params["with_original_language"] = language
    data = _get("/discover/movie", params)
    return data.get("results", []) if data else []


def enrich_movie(movie_id: int) -> dict:
    """Fetch and merge details + keywords + credits for one movie.
    The 3 independent calls run concurrently."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_details  = pool.submit(get_movie_details, movie_id)
        f_keywords = pool.submit(get_movie_keywords, movie_id)
        f_credits  = pool.submit(get_movie_credits, movie_id)
        details  = f_details.result() or {}
        keywords = f_keywords.result()
        credits  = f_credits.result()
    return {
        "tmdb_id":    movie_id,
        "title":      details.get("title", ""),
        "year":       (details.get("release_date") or "")[:4],
        "rating":     details.get("vote_average", 0),
        "popularity": details.get("popularity", 0),
        "genres":     [g["name"] for g in details.get("genres", [])],
        "overview":   details.get("overview", ""),
        "language":   details.get("original_language", "en"),
        "poster_url": get_poster_url(details.get("poster_path")),
        "keywords":   keywords,
        "director":   credits["director"],
        "cast":       credits["cast"],
    }
