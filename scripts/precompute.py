"""
Run once before launching the app:
    venv/bin/python scripts/precompute.py

Outputs to data/models/:
    user_movie_matrix.npz  — full rating matrix (users x movies)
    train_matrix.npz       — 80% chronological split
    meta.pkl               — index mappings + movie/user metadata
    funk_svd.pkl           — trained Funk SVD model (Surprise)
    item_cf.npz            — item-item similarity (K=20)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scipy.sparse as sp
import pickle

from data.loader import load_all
from data.preprocessor import (
    parse_movies,
    filter_sparse,
    build_user_movie_matrix,
    train_test_split_chronological,
)
from engines.collaborative import FunkSVDRecommender, ItemCFRecommender
from config import MODELS_DIR, CF_NEIGHBOURS

os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    print("Loading MovieLens 1M...")
    movies_df, ratings_df, users_df = load_all()
    print(f"  Raw: {len(ratings_df):,} ratings | {ratings_df['user_id'].nunique():,} users | {ratings_df['movie_id'].nunique():,} movies")

    print("Parsing movies...")
    movies_df = parse_movies(movies_df)

    print("Filtering sparse users/movies...")
    ratings_df = filter_sparse(ratings_df)
    print(f"  Clean: {len(ratings_df):,} ratings | {ratings_df['user_id'].nunique():,} users | {ratings_df['movie_id'].nunique():,} movies")

    print("Splitting train/test (80/20 chronological)...")
    train_df, test_df = train_test_split_chronological(ratings_df)
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    print("Building sparse matrices...")
    full_matrix, user_index, movie_index, index_user, index_movie = build_user_movie_matrix(ratings_df)
    train_matrix, _, _, _, _ = build_user_movie_matrix(train_df, user_index, movie_index)
    print(f"  Matrix shape: {full_matrix.shape} | Sparsity: {1 - full_matrix.nnz / (full_matrix.shape[0] * full_matrix.shape[1]):.1%}")

    print("Saving matrices + metadata...")
    sp.save_npz(f"{MODELS_DIR}/user_movie_matrix.npz", full_matrix)
    sp.save_npz(f"{MODELS_DIR}/train_matrix.npz", train_matrix)

    active_movie_ids = set(movie_index.keys())
    active_user_ids  = set(user_index.keys())

    with open(f"{MODELS_DIR}/meta.pkl", "wb") as f:
        pickle.dump({
            "user_index":  user_index,
            "movie_index": movie_index,
            "index_user":  index_user,
            "index_movie": index_movie,
            "movies_df":   movies_df[movies_df["movie_id"].isin(active_movie_ids)].copy(),
            "users_df":    users_df[users_df["user_id"].isin(active_user_ids)].copy(),
            "train_df":    train_df,
            "test_df":     test_df,
        }, f)

    print("\nTraining Funk SVD (n_factors=100, n_epochs=20) — takes ~3-5 min...")
    funk = FunkSVDRecommender()
    funk.fit(train_df, n_factors=100, n_epochs=20, lr=0.005, reg=0.02,
             save_path=f"{MODELS_DIR}/funk_svd.pkl")
    print("  Funk SVD training complete.")

    print(f"\nComputing Item-CF similarity (K={CF_NEIGHBOURS})...")
    item_cf = ItemCFRecommender()
    item_cf.fit(train_matrix, user_index, movie_index, K=CF_NEIGHBOURS,
                save_path=f"{MODELS_DIR}/item_cf.npz")
    print(f"  Item-CF shapes — nb_idx:{item_cf._nb_idx.shape}  nb_sims:{item_cf._nb_sims.shape}")

    print("\nDone. Files written:")
    for fname in ["user_movie_matrix.npz", "train_matrix.npz", "meta.pkl",
                  "funk_svd.pkl", "item_cf.npz"]:
        path = f"{MODELS_DIR}/{fname}"
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
