import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from config import CF_NEIGHBOURS, TOP_N_RECOMMENDATIONS, MODELS_DIR


class PopularityRecommender:
    def fit(self, ratings_df):
        self._user_watched = (
            ratings_df.groupby("user_id")["movie_id"].apply(set).to_dict()
        )
        self.movie_popularity = (
            ratings_df.groupby("movie_id")["rating"]
            .agg(count="count", mean="mean")
            .reset_index()
        )
        self.movie_popularity["score"] = (
            self.movie_popularity["count"] * self.movie_popularity["mean"]
        )
        self.movie_popularity = self.movie_popularity.sort_values("score", ascending=False)
        return self

    def recommend(self, user_id: int, n: int = TOP_N_RECOMMENDATIONS):
        watched = self._user_watched.get(user_id, set())
        candidates = self.movie_popularity[
            ~self.movie_popularity["movie_id"].isin(watched)
        ]
        return list(zip(candidates.head(n)["movie_id"].tolist(),
                        candidates.head(n)["score"].tolist()))


class ItemCFRecommender:
    def fit(self, train_matrix: sp.csr_matrix, user_index: dict, movie_index: dict,
            K: int = CF_NEIGHBOURS, save_path: str = None):
        self.user_index  = user_index
        self.movie_index = movie_index
        self.index_movie = {j: mid for mid, j in movie_index.items()}
        self._train_matrix = train_matrix
        K = min(K, train_matrix.shape[1] - 1)

        item_matrix = train_matrix.T.astype(np.float32)
        sim_matrix  = cosine_similarity(item_matrix)

        n_items = sim_matrix.shape[0]
        self._nb_idx  = np.zeros((n_items, K), dtype=np.int32)
        self._nb_sims = np.zeros((n_items, K), dtype=np.float32)

        for j in range(n_items):
            row = sim_matrix[j].copy()
            row[j] = -1.0
            top_k = np.argsort(row)[::-1][:K]
            self._nb_idx[j]  = top_k
            self._nb_sims[j] = row[top_k]

        if save_path:
            np.savez(save_path, nb_idx=self._nb_idx, nb_sims=self._nb_sims)
        return self

    def load(self, path: str, user_index: dict, movie_index: dict,
             train_matrix: sp.csr_matrix):
        data = np.load(path)
        self._nb_idx       = data["nb_idx"]
        self._nb_sims      = data["nb_sims"]
        self.user_index    = user_index
        self.movie_index   = movie_index
        self.index_movie   = {j: mid for mid, j in movie_index.items()}
        self._train_matrix = train_matrix
        return self

    def recommend(self, user_id: int, n: int = TOP_N_RECOMMENDATIONS):
        if user_id not in self.user_index:
            return []
        u_idx    = self.user_index[user_id]
        user_row = self._train_matrix.getrow(u_idx)
        if user_row.nnz == 0:
            return []

        rated_cols    = user_row.indices
        rated_ratings = user_row.data.astype(np.float32)
        seen          = set(rated_cols)
        score_num: dict[int, float] = {}
        score_den: dict[int, float] = {}

        for i_col, rating in zip(rated_cols, rated_ratings):
            for j_col, sim in zip(self._nb_idx[i_col], self._nb_sims[i_col]):
                if j_col in seen or sim <= 0:
                    continue
                score_num[j_col] = score_num.get(j_col, 0.0) + sim * rating
                score_den[j_col] = score_den.get(j_col, 0.0) + sim

        predictions = [
            (self.index_movie[j], float(np.clip(score_num[j] / score_den[j], 1.0, 5.0)))
            for j in score_num if score_den[j] > 0
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


class FunkSVDRecommender:
    """
    Regularised matrix factorisation via SGD — the algorithm from Simon Funk's
    Netflix Prize submission. Uses the Surprise library for training.
    Vectorised prediction: O(n_factors × n_items) per user, essentially instant.
    """

    def fit(self, train_df, n_factors: int = 100, n_epochs: int = 20,
            lr: float = 0.005, reg: float = 0.02, save_path: str = None):
        from surprise import SVD as SurpriseSVD, Dataset, Reader

        reader   = Reader(rating_scale=(1, 5))
        data     = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], reader)
        self.trainset = data.build_full_trainset()

        self.algo = SurpriseSVD(
            n_factors=n_factors, n_epochs=n_epochs,
            lr_all=lr, reg_all=reg, verbose=True,
        )
        self.algo.fit(self.trainset)

        # inner_iid → raw movie_id lookup (built once at fit time)
        self._inner_to_mid = {
            inner_iid: self.trainset.to_raw_iid(inner_iid)
            for inner_iid in range(self.trainset.n_items)
        }

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(self, f)
        return self

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def recommend(self, user_id: int, n: int = TOP_N_RECOMMENDATIONS,
                  allowed_movie_ids: set = None):
        """SVD top-N. If allowed_movie_ids is given, candidates are restricted
        to that set first (e.g. a persona's signature genres) and then ranked
        by predicted rating — a genre-filter + collaborative-ranking hybrid."""
        try:
            u_inner = self.trainset.to_inner_uid(user_id)
        except ValueError:
            return []

        seen_inner = {iid for iid, _ in self.trainset.ur[u_inner]}

        p_u = self.algo.pu[u_inner]
        b_u = self.algo.bu[u_inner]
        mu  = self.trainset.global_mean

        # vectorised: predict all items at once
        scores = mu + b_u + self.algo.bi + self.algo.qi @ p_u

        predictions = [
            (mid, float(np.clip(scores[i_inner], 1.0, 5.0)))
            for i_inner in range(len(scores))
            if i_inner not in seen_inner
            and (mid := self._inner_to_mid[i_inner]) is not None
            and (allowed_movie_ids is None or mid in allowed_movie_ids)
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    def predict_single(self, user_id: int, movie_id: int) -> float:
        try:
            pred = self.algo.predict(user_id, movie_id)
            return float(np.clip(pred.est, 1.0, 5.0))
        except Exception:
            return self.trainset.global_mean
