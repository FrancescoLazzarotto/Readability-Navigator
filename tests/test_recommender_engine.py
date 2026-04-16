import numpy as np
import pandas as pd

from src.recommender.recommender_engine import RecommenderEngine


def _build_engine() -> RecommenderEngine:
    df = pd.DataFrame(
        {
            "id": ["d1", "d2", "d3"],
            "testo": ["harder", "easier", "outside_tol"],
            "flesch_score": [50.0, 70.0, 85.0],
        }
    )

    embedding = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    config = {"k": 2, "eta": 0.6, "zeta": 0.4, "alpha": 0.4, "tol": 20}
    return RecommenderEngine(df=df, embedding=embedding, config=config, user_id=None, profile_path=None)


def test_catalog_applies_history_and_tolerance() -> None:
    engine = _build_engine()

    user = {
        "user_id": 1,
        "topic_vector": [1.0, 0.0, 0.0],
        "target_readability": 60,
        "history": ["d2"],
    }

    catalog = engine.catalog(user)
    assert catalog["id"].tolist() == ["d1"]


def test_hybrid_ranking_penalizes_harder_text_more() -> None:
    engine = _build_engine()

    user = {
        "user_id": 1,
        "topic_vector": [1.0, 0.0, 0.0],
        "target_readability": 60,
        "history": [],
    }

    titles, scores, _, _ = engine.rank_top_k(user, mode="hybrid")

    # d1 and d2 have same semantic similarity and same absolute gap (10),
    # but d1 is harder (50 < 60) so it gets asymmetric penalty.
    assert titles[0] == "d2"
    assert scores[0] > scores[1]
