import pandas as pd

from src.eval.evaluation import RecommenderEvaluation
from src.recommender.models import RecommendationItem, RecommenderConfig


class DummyEngine:
    def __init__(self) -> None:
        self.config = RecommenderConfig(k=3, eta=0.6, zeta=0.4, alpha=0.4, tol=15)

    def rank_top_k_items(self, user, mode="hybrid"):
        if mode == "similarity_only":
            return [
                RecommendationItem("a", 0.95, "A", 80.0, 0.95, 0.0, 1.0),
                RecommendationItem("b", 0.90, "B", 30.0, 0.90, 0.0, 1.0),
                RecommendationItem("c", 0.10, "C", 60.0, 0.10, 0.0, 1.0),
            ]

        if mode == "readability_only":
            return [
                RecommendationItem("a", 0.95, "A", 60.0, 0.10, 0.0, 1.0),
                RecommendationItem("b", 0.90, "B", 62.0, 0.10, 0.0, 1.0),
                RecommendationItem("c", 0.10, "C", 30.0, 0.10, 0.0, 1.0),
            ]

        return [
            RecommendationItem("a", 0.95, "A", 60.0, 0.90, 0.0, 1.0),
            RecommendationItem("b", 0.90, "B", 62.0, 0.80, 0.0, 1.0),
            RecommendationItem("c", 0.10, "C", 30.0, 0.10, 0.0, 1.0),
        ]


def test_evaluate_user_returns_metrics_in_unit_interval() -> None:
    evaluator = RecommenderEvaluation(k=3)
    engine = DummyEngine()
    user = {"user_id": 1, "target_readability": 60}

    result = evaluator.evaluate_user(engine, user, mode="hybrid")

    assert 0.0 <= result.ndcg_at_k <= 1.0
    assert 0.0 <= result.precision_at_k <= 1.0
    assert 0.0 <= result.precision_at_5 <= 1.0
    assert 0.0 <= result.mean_cos_at_5 <= 1.0
    assert result.hndcg_cos <= 1.0
    assert result.hp_cos <= 1.0


def test_run_ablation_returns_three_methods() -> None:
    evaluator = RecommenderEvaluation(k=3)
    engine = DummyEngine()
    users = [{"user_id": 1, "target_readability": 60}]

    ablation = evaluator.run_ablation(engine, users)

    assert isinstance(ablation, pd.DataFrame)
    assert set(ablation["Method"].tolist()) == {
        "Readability only",
        "Similarity only",
        "Hybrid",
    }
