from __future__ import annotations

from dataclasses import dataclass

from src.recommender.models import RecommenderConfig, ScoringBreakdown


@dataclass
class HybridScoringModel:
    """Implements the hybrid scoring function defined in the paper."""

    config: RecommenderConfig

    @staticmethod
    def asymmetric_penalty(target_readability: float, doc_readability: float, alpha: float) -> float:
        """Apply higher penalty only when the text is harder than the user target.

        In Flesch Reading Ease, lower values mean harder texts.
        """

        return 1.0 + alpha if doc_readability < target_readability else 1.0

    def normalized_gap(self, target_readability: float, doc_readability: float) -> float:
        """Local readability-gap normalization from the paper.

        G_tilde(u, d) = min(|tau_u - r_d| / tol, 1)
        """

        raw_gap = abs(target_readability - doc_readability)
        return min(raw_gap / self.config.tol, 1.0)

    def score(self, semantic_similarity: float, target_readability: float, doc_readability: float) -> ScoringBreakdown:
        """Compute the hybrid objective:

        S(u, d) = eta * cos(v_u, e_d) - zeta * G_tilde(u, d) * P(u, d)
        """

        gap = self.normalized_gap(target_readability, doc_readability)
        penalty = self.asymmetric_penalty(target_readability, doc_readability, self.config.alpha)
        final_score = self.config.eta * semantic_similarity - self.config.zeta * gap * penalty

        return ScoringBreakdown(
            score=float(final_score),
            semantic_similarity=float(semantic_similarity),
            normalized_gap=float(gap),
            penalty=float(penalty),
        )

    def score_similarity_only(self, semantic_similarity: float) -> ScoringBreakdown:
        """Ablation mode that uses only semantic relevance."""

        return ScoringBreakdown(
            score=float(semantic_similarity),
            semantic_similarity=float(semantic_similarity),
            normalized_gap=0.0,
            penalty=1.0,
        )

    def score_readability_only(self, target_readability: float, doc_readability: float) -> ScoringBreakdown:
        """Ablation mode that uses only the readability-alignment term."""

        gap = self.normalized_gap(target_readability, doc_readability)
        penalty = self.asymmetric_penalty(target_readability, doc_readability, self.config.alpha)
        score = 1.0 - gap * penalty

        return ScoringBreakdown(
            score=float(score),
            semantic_similarity=0.0,
            normalized_gap=float(gap),
            penalty=float(penalty),
        )
