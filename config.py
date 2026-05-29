import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_IMAGE_PLACEHOLDER = "https://via.placeholder.com/500x750?text=No+Poster"

DATA_DIR = "data/raw"
MODELS_DIR = "data/models"
TMDB_CACHE_DIR = "data/tmdb_cache"
DB_PATH = "storage/recommendations.db"

MOVIES_FILE = f"{DATA_DIR}/movies.dat"
RATINGS_FILE = f"{DATA_DIR}/ratings.dat"
USERS_FILE = f"{DATA_DIR}/users.dat"

MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 10
CF_NEIGHBOURS = 20
TOP_N_RECOMMENDATIONS = 10

INDIAN_LANGUAGE_CODES = ["hi", "ta", "te", "ml", "kn"]

# Curated MovieLens users with clear, distinct tastes — shown in the Watch
# History tab so demo viewers understand whose profile they're exploring.
# user_id chosen by analysing each user's rating history; `genres` are the
# signature genres used to filter SVD recommendations so results visibly
# match the persona (SVD still ranks the best film within the genre).
WATCH_HISTORY_PERSONAS = [
    {"label": "Action Buff",        "user_id": 2886, "genres": ["Action", "Adventure"],     "desc": "Action, war & adventure"},
    {"label": "Rom-Com Lover",      "user_id": 927,  "genres": ["Romance", "Comedy"],       "desc": "Comedy & romance"},
    {"label": "Sci-Fi Geek",        "user_id": 2167, "genres": ["Sci-Fi"],                  "desc": "Science fiction"},
    {"label": "Horror Fan",         "user_id": 5597, "genres": ["Horror"],                  "desc": "Horror & slashers"},
    {"label": "Family / Animation", "user_id": 1353, "genres": ["Animation", "Children's"], "desc": "Animation & family"},
    {"label": "Classic Cinephile",  "user_id": 3039, "genres": ["Drama", "Film-Noir"],      "desc": "Golden-age drama"},
]

TMDB_GENRE_IDS = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
    "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
    "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
    "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
    "Sci-Fi": 878, "Thriller": 53, "War": 10752, "Western": 37,
}

# reverse map: TMDB genre id → name (for candidates fetched via search/discover,
# which return genre_ids rather than genre names — avoids per-movie enrich calls)
TMDB_GENRE_NAMES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Sci-Fi", 10770: "TV Movie",
    53: "Thriller", 10752: "War", 37: "Western",
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MOOD_KEYWORD_MAP = {
    "Intense": ["action", "thriller", "war", "crime", "revenge"],
    "Light": ["comedy", "romance", "animation", "family", "musical"],
    "Mind-bending": ["sci-fi", "mystery", "psychological", "twist", "surreal"],
    "Emotional": ["drama", "tragedy", "love", "loss", "grief"],
}
