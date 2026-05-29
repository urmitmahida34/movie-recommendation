"""
Run after precompute.py:
    venv/bin/python scripts/evaluate.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import scipy.sparse as sp
import numpy as np

from engines.collaborative import FunkSVDRecommender
from engines.evaluator import evaluate_model
from config import MODELS_DIR

print("Loading precomputed data...")
with open(f"{MODELS_DIR}/meta.pkl", "rb") as f:
    meta = pickle.load(f)

train_matrix = sp.load_npz(f"{MODELS_DIR}/train_matrix.npz")
test_df      = meta["test_df"]
print(f"  Test set: {len(test_df):,} ratings across {test_df['user_id'].nunique():,} users\n")

print("Loading Funk SVD model...")
funk = FunkSVDRecommender.load(f"{MODELS_DIR}/funk_svd.pkl")

print("Evaluating Funk SVD (Precision@10, Recall@10, NDCG@10)...")
results = evaluate_model(funk, test_df, k=10)
print(f"  {results}")

# RMSE via Surprise's built-in accuracy
print("\nComputing RMSE on test set (sample 10k)...")
sample = test_df.sample(min(10000, len(test_df)), random_state=42)
actuals, estimates = [], []
for row in sample.itertuples():
    # skip users/items not seen in training (matches old "was_impossible" skip)
    if row.user_id in funk.user_inner and row.movie_id in funk.mid_to_inner:
        actuals.append(row.rating)
        estimates.append(funk.predict_single(row.user_id, row.movie_id))
rmse = float(np.sqrt(np.mean((np.array(actuals) - np.array(estimates)) ** 2)))
print(f"  RMSE: {rmse:.4f}")

print("\n" + "=" * 55)
print(f"{'Metric':<20} {'Value':>10}")
print("-" * 55)
k = 10
for metric, val in results.items():
    if metric != "users_evaluated":
        print(f"{metric:<20} {val:>10.4f}")
print(f"{'RMSE':<20} {rmse:>10.4f}")
print(f"{'users_evaluated':<20} {results['users_evaluated']:>10}")
print("=" * 55)
