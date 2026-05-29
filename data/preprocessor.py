import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from config import MIN_USER_RATINGS, MIN_MOVIE_RATINGS


def parse_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    def _split_title_year(raw):
        match = re.match(r"^(.*)\((\d{4})\)\s*$", raw.strip())
        if match:
            return match.group(1).strip(), int(match.group(2))
        return raw.strip(), None

    titles, years = zip(*movies_df["title_raw"].map(_split_title_year))
    movies_df = movies_df.copy()
    movies_df["title"] = titles
    movies_df["year"] = years
    movies_df["genres"] = movies_df["genres_raw"].str.split("|")
    return movies_df.drop(columns=["title_raw", "genres_raw"])


def filter_sparse(ratings_df: pd.DataFrame) -> pd.DataFrame:
    # keep users with enough ratings
    user_counts = ratings_df["user_id"].value_counts()
    active_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    ratings_df = ratings_df[ratings_df["user_id"].isin(active_users)]

    # keep movies with enough ratings
    movie_counts = ratings_df["movie_id"].value_counts()
    active_movies = movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index
    ratings_df = ratings_df[ratings_df["movie_id"].isin(active_movies)]

    return ratings_df.reset_index(drop=True)


def build_user_movie_matrix(ratings_df: pd.DataFrame,
                             user_index: dict = None,
                             movie_index: dict = None):
    """
    Returns:
        matrix      — scipy CSR sparse matrix (users x movies)
        user_index  — dict mapping user_id → row index
        movie_index — dict mapping movie_id → col index
        index_user  — dict mapping row index → user_id
        index_movie — dict mapping col index → movie_id

    Pass existing user_index / movie_index to share the same index space
    across full / train / test matrices (avoids row-column mismatches).
    """
    if user_index is None:
        users = sorted(ratings_df["user_id"].unique())
        user_index = {uid: i for i, uid in enumerate(users)}
    if movie_index is None:
        movies = sorted(ratings_df["movie_id"].unique())
        movie_index = {mid: j for j, mid in enumerate(movies)}

    index_user  = {i: uid for uid, i in user_index.items()}
    index_movie = {j: mid for mid, j in movie_index.items()}

    rows = ratings_df["user_id"].map(user_index).values
    cols = ratings_df["movie_id"].map(movie_index).values
    data = ratings_df["rating"].astype(np.float32).values

    matrix = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_index), len(movie_index)),
        dtype=np.float32,
    )
    return matrix, user_index, movie_index, index_user, index_movie


def train_test_split_chronological(ratings_df: pd.DataFrame, test_ratio: float = 0.2):
    """Chronological split per user — last test_ratio% of each user's ratings go to test."""
    ratings_df = ratings_df.sort_values(["user_id", "timestamp"])
    train_rows, test_rows = [], []

    for _, group in ratings_df.groupby("user_id"):
        n = len(group)
        cutoff = int(n * (1 - test_ratio))
        train_rows.append(group.iloc[:cutoff])
        test_rows.append(group.iloc[cutoff:])

    train = pd.concat(train_rows).reset_index(drop=True)
    test = pd.concat(test_rows).reset_index(drop=True)
    return train, test
