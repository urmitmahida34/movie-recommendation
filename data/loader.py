import pandas as pd
from config import MOVIES_FILE, RATINGS_FILE, USERS_FILE


def load_movies() -> pd.DataFrame:
    return pd.read_csv(
        MOVIES_FILE,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movie_id", "title_raw", "genres_raw"],
    )


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(
        RATINGS_FILE,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )


def load_users() -> pd.DataFrame:
    return pd.read_csv(
        USERS_FILE,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "gender", "age", "occupation", "zip"],
    )


def load_all():
    return load_movies(), load_ratings(), load_users()
