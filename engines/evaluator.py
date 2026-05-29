import numpy as np
import pandas as pd


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, r in enumerate(top_k)
        if r in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(recommender, test_df: pd.DataFrame, k: int = 10,
                   min_relevant: int = 1, rating_threshold: float = 4.0,
                   sample_users: int = 500, seed: int = 42) -> dict:
    """
    Evaluates a recommender on test_df.
    recommender must have a .recommend(user_id) method returning [(movie_id, score), ...]
    """
    rng = np.random.default_rng(seed)
    users = test_df["user_id"].unique()
    if len(users) > sample_users:
        users = rng.choice(users, size=sample_users, replace=False)

    precisions, recalls, ndcgs = [], [], []

    for user_id in users:
        user_test = test_df[test_df["user_id"] == user_id]
        relevant = set(
            user_test[user_test["rating"] >= rating_threshold]["movie_id"].tolist()
        )
        if len(relevant) < min_relevant:
            continue

        recs = recommender.recommend(user_id, n=k)
        if not recs:
            continue
        rec_ids = [mid for mid, _ in recs]

        precisions.append(precision_at_k(rec_ids, relevant, k))
        recalls.append(recall_at_k(rec_ids, relevant, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k))

    return {
        f"precision@{k}": round(np.mean(precisions), 4) if precisions else 0.0,
        f"recall@{k}":    round(np.mean(recalls), 4)    if recalls    else 0.0,
        f"ndcg@{k}":      round(np.mean(ndcgs), 4)      if ndcgs      else 0.0,
        "users_evaluated": len(precisions),
    }
