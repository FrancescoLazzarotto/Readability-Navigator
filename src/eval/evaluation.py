from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import ndcg_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.recommender.recommender_engine import RecommenderEngine
from utils.io_utils import load_yaml


@dataclass(frozen=True)
class EvaluationWeights:
    """Weights used to combine semantic and readability relevance signals."""

    semantic: float = 0.5
    readability: float = 0.5
    relevance_threshold: float = 0.6

    def normalized(self) -> "EvaluationWeights":
        total = self.semantic + self.readability
        if total <= 0:
            raise ValueError("semantic + readability weights must be > 0")
        return EvaluationWeights(
            semantic=self.semantic / total,
            readability=self.readability / total,
            relevance_threshold=self.relevance_threshold,
        )


@dataclass(frozen=True)
class UserEvaluationResult:
    """Offline evaluation metrics for one user profile."""

    user_id: str
    ndcg_at_k: float
    precision_at_k: float
    precision_at_5: float
    mean_cos_at_5: float
    hndcg_cos: float
    hp_cos: float


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the stratified simulated-user protocol."""

    seeds: int = 10
    n_topic_clusters: int = 8
    readability_anchors: tuple[int, int, int, int] = (30, 45, 60, 75)
    users_per_cell: int = 4
    history_size: int = 5
    readability_noise_std: float = 2.5
    vector_noise_std: float = 0.01
    readability_min: float = 20.0
    readability_max: float = 90.0


class SimulatedUserGenerator:
    """Generate stratified synthetic users as described in the paper."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def generate_users(
        self,
        df: pd.DataFrame,
        embedding: Sequence[Sequence[float]],
        seed: int,
    ) -> List[Dict[str, object]]:
        """Generate one seed of stratified simulated user profiles."""

        rng = np.random.default_rng(seed)

        emb = np.asarray(embedding, dtype=float)
        row_norms = np.linalg.norm(emb, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        emb_normalized = emb / row_norms

        n_clusters = min(self.config.n_topic_clusters, len(df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(emb_normalized)

        ids = df["id"].astype(str).to_numpy()
        flesch = df["flesch_score"].astype(float).to_numpy()

        users: List[Dict[str, object]] = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            for anchor in self.config.readability_anchors:
                for user_idx in range(self.config.users_per_cell):
                    sampled_target = float(
                        np.clip(
                            anchor + rng.normal(0, self.config.readability_noise_std),
                            self.config.readability_min,
                            self.config.readability_max,
                        )
                    )

                    sorted_by_gap = cluster_indices[
                        np.argsort(np.abs(flesch[cluster_indices] - sampled_target))
                    ]
                    history_size = min(self.config.history_size, len(sorted_by_gap))

                    candidate_pool_size = min(len(sorted_by_gap), max(history_size * 2, history_size))
                    candidate_pool = sorted_by_gap[:candidate_pool_size]

                    history_indices = (
                        rng.choice(candidate_pool, size=history_size, replace=False)
                        if history_size > 0
                        else np.asarray([], dtype=int)
                    )

                    if history_size > 0:
                        centroid = emb_normalized[history_indices].mean(axis=0)
                    else:
                        centroid = emb_normalized[cluster_indices].mean(axis=0)

                    noise = rng.normal(0, self.config.vector_noise_std, size=centroid.shape)
                    topic_vector = self._normalize(centroid + noise)

                    users.append(
                        {
                            "user_id": f"sim_s{seed}_c{cluster_id}_a{anchor}_u{user_idx}",
                            "topic_vector": topic_vector.tolist(),
                            "target_readability": sampled_target,
                            "history": [ids[idx] for idx in history_indices],
                        }
                    )

        return users


class RecommenderEvaluation:
    """Offline multi-objective recommender evaluation aligned with the paper."""

    def __init__(
        self,
        k: int,
        *,
        mean_cos_k: int = 5,
        weights: EvaluationWeights | None = None,
    ) -> None:
        self.k = int(k)
        self.mean_cos_k = int(mean_cos_k)
        self.weights = (weights or EvaluationWeights()).normalized()
        self.ndcg_history: List[float] = []
        self.results: List[UserEvaluationResult] = []

    @staticmethod
    def harmonic_mean(a: float, b: float) -> float:
        """Compute harmonic mean while safely handling zero values."""

        if a <= 0 or b <= 0:
            return 0.0
        return float((2 * a * b) / (a + b))

    @staticmethod
    def _to_unit_interval_cos(values: Iterable[float]) -> np.ndarray:
        """Map cosine scores from [-1, 1] to [0, 1]."""

        arr = np.asarray(list(values), dtype=float)
        return np.clip((arr + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def compute_readability_alignment(
        flesch_scores: Iterable[float],
        target_readability: float,
        tol: float,
    ) -> np.ndarray:
        """Compute readability alignment using the local normalized gap."""

        flesch = np.asarray(list(flesch_scores), dtype=float)
        gaps = np.abs(flesch - float(target_readability))
        normalized_gap = np.minimum(gaps / float(tol), 1.0)
        return 1.0 - normalized_gap

    def compute_joint_relevance(
        self,
        semantic_scores: Iterable[float],
        flesch_scores: Iterable[float],
        target_readability: float,
        tol: float,
    ) -> np.ndarray:
        """Combine semantic and readability relevance into a single target signal."""

        semantic = self._to_unit_interval_cos(semantic_scores)
        readability = self.compute_readability_alignment(
            flesch_scores=flesch_scores,
            target_readability=target_readability,
            tol=tol,
        )
        return self.weights.semantic * semantic + self.weights.readability * readability

    def ndcg_at_k(self, relevance: np.ndarray, pred_scores: Sequence[float]) -> float:
        """Compute NDCG@K."""

        if len(relevance) == 0:
            return 0.0
        y_true = relevance.reshape(1, -1)
        y_score = np.asarray(pred_scores, dtype=float).reshape(1, -1)
        return float(ndcg_score(y_true, y_score, k=self.k))

    def precision_at_k(self, relevance: np.ndarray, pred_scores: Sequence[float], k: int) -> float:
        """Compute Precision@K using joint relevance thresholding."""

        if len(relevance) == 0:
            return 0.0
        k_eff = min(int(k), len(relevance))
        top_idx = np.argsort(np.asarray(pred_scores, dtype=float))[::-1][:k_eff]
        top_relevance = relevance[top_idx]
        return float(np.mean(top_relevance >= self.weights.relevance_threshold))

    def mean_cos_at_k(
        self,
        semantic_scores: Sequence[float],
        pred_scores: Sequence[float],
        k: int,
    ) -> float:
        """Compute MeanCos@K on ranked results."""

        if len(semantic_scores) == 0:
            return 0.0
        k_eff = min(int(k), len(semantic_scores))
        top_idx = np.argsort(np.asarray(pred_scores, dtype=float))[::-1][:k_eff]
        semantic = np.clip(np.asarray(semantic_scores, dtype=float), 0.0, 1.0)
        return float(np.mean(semantic[top_idx]))

    def evaluate_user(self, recommender: RecommenderEngine, user: Dict[str, object], mode: str = "hybrid") -> UserEvaluationResult:
        """Evaluate one user profile under one ranking mode."""

        items = recommender.rank_top_k_items(user, mode=mode)
        if not items:
            return UserEvaluationResult(
                user_id=str(user.get("user_id", "unknown")),
                ndcg_at_k=0.0,
                precision_at_k=0.0,
                precision_at_5=0.0,
                mean_cos_at_5=0.0,
                hndcg_cos=0.0,
                hp_cos=0.0,
            )

        pred_scores = np.asarray([item.score for item in items], dtype=float)
        semantic_scores = np.asarray([item.semantic_similarity for item in items], dtype=float)
        flesch_scores = np.asarray([item.flesch_score for item in items], dtype=float)

        relevance = self.compute_joint_relevance(
            semantic_scores=semantic_scores,
            flesch_scores=flesch_scores,
            target_readability=float(user["target_readability"]),
            tol=recommender.config.tol,
        )

        ndcg_k = self.ndcg_at_k(relevance, pred_scores)
        precision_k = self.precision_at_k(relevance, pred_scores, self.k)
        precision_5 = self.precision_at_k(relevance, pred_scores, 5)
        mean_cos_5 = self.mean_cos_at_k(semantic_scores, pred_scores, self.mean_cos_k)

        return UserEvaluationResult(
            user_id=str(user.get("user_id", "unknown")),
            ndcg_at_k=ndcg_k,
            precision_at_k=precision_k,
            precision_at_5=precision_5,
            mean_cos_at_5=mean_cos_5,
            hndcg_cos=self.harmonic_mean(ndcg_k, mean_cos_5),
            hp_cos=self.harmonic_mean(precision_5, mean_cos_5),
        )

    def evaluate_users(
        self,
        recommender: RecommenderEngine,
        users: Sequence[Dict[str, object]],
        mode: str = "hybrid",
    ) -> float:
        """Backward-compatible entrypoint returning mean NDCG@K."""

        self.results = [self.evaluate_user(recommender, user, mode=mode) for user in users]
        self.ndcg_history = [result.ndcg_at_k for result in self.results]
        return float(np.mean(self.ndcg_history)) if self.ndcg_history else 0.0

    def results_to_dataframe(self) -> pd.DataFrame:
        """Return detailed user-level metrics as a dataframe."""

        return pd.DataFrame([result.__dict__ for result in self.results])

    def summarize(self) -> Dict[str, float]:
        """Return mean metrics over current user-level results."""

        if not self.results:
            return {
                "NDCG@K": 0.0,
                "Precision@K": 0.0,
                "Precision@5": 0.0,
                "MeanCos@5": 0.0,
                "HNDCG-Cos": 0.0,
                "HP-Cos": 0.0,
            }

        df = self.results_to_dataframe()
        return {
            "NDCG@K": float(df["ndcg_at_k"].mean()),
            "Precision@K": float(df["precision_at_k"].mean()),
            "Precision@5": float(df["precision_at_5"].mean()),
            "MeanCos@5": float(df["mean_cos_at_5"].mean()),
            "HNDCG-Cos": float(df["hndcg_cos"].mean()),
            "HP-Cos": float(df["hp_cos"].mean()),
        }

    def run_ablation(
        self,
        recommender: RecommenderEngine,
        users: Sequence[Dict[str, object]],
    ) -> pd.DataFrame:
        """Evaluate readability-only, similarity-only and hybrid ranking variants."""

        rows = []
        mode_map = {
            "Readability only": "readability_only",
            "Similarity only": "similarity_only",
            "Hybrid": "hybrid",
        }
        for label, mode in mode_map.items():
            self.evaluate_users(recommender, users, mode=mode)
            summary = self.summarize()
            summary["Method"] = label
            rows.append(summary)

        return pd.DataFrame(rows)[
            [
                "Method",
                "NDCG@K",
                "Precision@K",
                "Precision@5",
                "MeanCos@5",
                "HNDCG-Cos",
                "HP-Cos",
            ]
        ]

    def run_simulated_protocol(
        self,
        df: pd.DataFrame,
        embedding: Sequence[Sequence[float]],
        recommender_config: Dict[str, float | int],
        simulation_config: SimulationConfig | None = None,
    ) -> pd.DataFrame:
        """Run seed-based simulated protocol and report mean +- std per method."""

        sim_config = simulation_config or SimulationConfig()
        generator = SimulatedUserGenerator(sim_config)

        seed_tables: List[pd.DataFrame] = []
        for seed in range(sim_config.seeds):
            users = generator.generate_users(df=df, embedding=embedding, seed=seed)
            recommender = RecommenderEngine(
                df=df,
                embedding=embedding,
                config=recommender_config,
                user_id=None,
                profile_path=None,
            )
            seed_table = self.run_ablation(recommender, users)
            seed_table["seed"] = seed
            seed_tables.append(seed_table)

        all_results = pd.concat(seed_tables, ignore_index=True)
        metric_columns = [
            "NDCG@K",
            "Precision@K",
            "Precision@5",
            "MeanCos@5",
            "HNDCG-Cos",
            "HP-Cos",
        ]

        summary_rows = []
        for method, method_df in all_results.groupby("Method", sort=False):
            row: Dict[str, str] = {"Method": str(method)}
            for metric in metric_columns:
                mean_value = float(method_df[metric].mean())
                std_value = float(method_df[metric].std(ddof=0))
                row[metric] = f"{mean_value:.4f} +- {std_value:.4f}"
            summary_rows.append(row)

        return pd.DataFrame(summary_rows)[["Method", *metric_columns]]


if __name__ == "__main__":
    from utils.data_loader import load_embedding, load_features_df

    config = load_yaml()
    configuration = {
        "tol": config["tol"],
        "eta": config["eta"],
        "zeta": config["zeta"],
        "alpha": config["alpha"],
        "k": config["k"],
    }

    df = load_features_df()
    embedding = load_embedding()

    rel_profile_path = config["paths"]["user_json"]
    profile_path = os.path.join(PROJECT_ROOT, rel_profile_path)
    users = []
    for filename in os.listdir(profile_path):
        if filename.endswith(".json"):
            with open(os.path.join(profile_path, filename), "r", encoding="utf-8") as file:
                users.append(json.load(file))

    recommender = RecommenderEngine(
        df=df,
        embedding=embedding,
        config=configuration,
        user_id=None,
        profile_path=profile_path,
    )

    evaluator = RecommenderEvaluation(k=configuration["k"])
    ablation_df = evaluator.run_ablation(recommender, users)

    print("Offline ablation summary")
    print(ablation_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if os.getenv("RUN_SIMULATED_PROTOCOL", "0") == "1":
        simulated_df = evaluator.run_simulated_protocol(
            df=df,
            embedding=embedding,
            recommender_config=configuration,
        )
        print("\nSimulated protocol summary (mean +- std across seeds)")
        print(simulated_df.to_string(index=False))
