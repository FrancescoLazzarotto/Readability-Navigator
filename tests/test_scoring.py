import pytest

from src.recommender.models import RecommenderConfig
from src.recommender.scoring import HybridScoringModel


def test_local_gap_normalization_clips_at_one() -> None:
    config = RecommenderConfig(k=10, eta=0.6, zeta=0.4, alpha=0.4, tol=15)
    scorer = HybridScoringModel(config)

    assert scorer.normalized_gap(target_readability=60, doc_readability=90) == 1.0


def test_asymmetric_penalty_targets_harder_texts() -> None:
    config = RecommenderConfig(k=10, eta=0.6, zeta=0.4, alpha=0.4, tol=15)
    scorer = HybridScoringModel(config)

    harder_text_penalty = scorer.asymmetric_penalty(60, 55, alpha=config.alpha)
    easier_text_penalty = scorer.asymmetric_penalty(60, 70, alpha=config.alpha)

    assert harder_text_penalty == pytest.approx(1.4)
    assert easier_text_penalty == pytest.approx(1.0)


def test_hybrid_scoring_matches_paper_equation() -> None:
    config = RecommenderConfig(k=10, eta=0.6, zeta=0.4, alpha=0.4, tol=15)
    scorer = HybridScoringModel(config)

    result = scorer.score(
        semantic_similarity=0.8,
        target_readability=60,
        doc_readability=55,
    )

    expected_gap = min(abs(60 - 55) / 15, 1.0)
    expected_penalty = 1.4
    expected_score = 0.6 * 0.8 - 0.4 * expected_gap * expected_penalty

    assert result.normalized_gap == pytest.approx(expected_gap)
    assert result.penalty == pytest.approx(expected_penalty)
    assert result.score == pytest.approx(expected_score)
