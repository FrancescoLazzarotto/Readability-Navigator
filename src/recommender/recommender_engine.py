from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.recommender.models import RecommenderConfig, RecommendationItem, UserProfile
from src.recommender.scoring import HybridScoringModel
from src.user.model_user import load_user_model


class RecommenderEngine:
    """Recommendation engine.

    The engine is organized into three steps:
    1. Candidate generation with readability tolerance and history exclusion.
    2. Hybrid scoring that balances semantic relevance and cognitive load.
    3. Top-K ranking.
    """

    REQUIRED_COLUMNS = ("id", "testo", "flesch_score")

    def __init__(
        self,
        df: pd.DataFrame,
        embedding: Sequence[Sequence[float]],
        config: Optional[Dict[str, Any]],
        user_id: Optional[str],
        profile_path: Optional[str],
    ) -> None:
        self.config = RecommenderConfig.from_dict(config)
        self.scorer = HybridScoringModel(self.config)
        self.user_id = user_id
        self.profile_path = profile_path

        self.df = df.copy().reset_index(drop=True)
        missing = [column for column in self.REQUIRED_COLUMNS if column not in self.df.columns]
        if missing:
            raise ValueError(f"Dataframe is missing required columns: {missing}")

        self.df["id"] = self.df["id"].astype(str)
        if self.df["id"].duplicated().any():
            raise ValueError("Document IDs must be unique for deterministic ranking")

        self.embedding = np.asarray(embedding, dtype=float)
        if self.embedding.ndim != 2:
            raise ValueError("embedding must be a 2D array-like structure")
        if len(self.embedding) != len(self.df):
            raise ValueError("embedding size must match number of documents in dataframe")

        self._id_to_index = {
            doc_id: idx for idx, doc_id in enumerate(self.df["id"].tolist())
        }

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity with safe zero-norm handling."""

        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)

    def _coerce_user_profile(self, user: Any) -> UserProfile:
        """Accept either dict-based or dataclass-based user profiles."""

        if isinstance(user, UserProfile):
            return user
        if isinstance(user, dict):
            return UserProfile.from_dict(user)
        raise TypeError("user must be either a dict or UserProfile")

    def profile(self) -> Optional[dict]:
        """Load user profile from JSON if available."""

        if self.user_id is None or self.profile_path is None:
            return None
        try:
            filename = f"user{self.user_id}.json"
            return load_user_model(filename, self.profile_path)
        except FileNotFoundError:
            return None

    def catalog(self, profile: Any) -> pd.DataFrame:
        """Build candidate set C_t(u) = {d not in H_t and |r_d - tau_t| <= tol}."""

        user_profile = self._coerce_user_profile(profile)
        filtered = self.df[~self.df["id"].isin(user_profile.history)]
        filtered = filtered[
            np.abs(filtered["flesch_score"] - user_profile.target_readability)
            <= self.config.tol
        ]
        return filtered

    def _resolve_doc_index(self, doc_id: str) -> int:
        doc_key = str(doc_id)
        if doc_key not in self._id_to_index:
            raise ValueError(f"Document not found: {doc_key}")
        return self._id_to_index[doc_key]

    def get_document(self, doc_id: str) -> Tuple[str, np.ndarray]:
        """Return document text and embedding by document ID."""

        idx = self._resolve_doc_index(str(doc_id))
        text = str(self.df.at[idx, "testo"])
        emb = self.embedding[idx].reshape(-1)
        return text, emb

    def get_flesch(self, doc_id: str) -> float:
        """Return Flesch Reading Ease score by document ID."""

        idx = self._resolve_doc_index(str(doc_id))
        return float(self.df.at[idx, "flesch_score"])

    def gap_readability(self, user: Any, flesch: float) -> Tuple[float, float, float]:
        """Return readability gap, user target readability and document readability."""

        user_profile = self._coerce_user_profile(user)
        target = user_profile.target_readability
        gap = abs(target - float(flesch))
        return float(gap), float(target), float(flesch)

    def penalty(self, target: float, readability: float, alpha: Optional[float] = None) -> float:
        """Compatibility wrapper for the asymmetric penalty rule from the paper."""

        alpha_value = self.config.alpha if alpha is None else float(alpha)
        return self.scorer.asymmetric_penalty(target, readability, alpha_value)

    def theme_similarity(self, user: Any, doc_id: str) -> float:
        """Compute cosine similarity between user topic vector and document embedding."""

        user_profile = self._coerce_user_profile(user)
        _, doc_embedding = self.get_document(doc_id)
        return self._cosine_similarity(user_profile.topic_vector, doc_embedding)

    def _build_item(self, user_profile: UserProfile, doc_id: str, mode: str) -> RecommendationItem:
        idx = self._resolve_doc_index(str(doc_id))
        doc_embedding = self.embedding[idx].reshape(-1)
        doc_readability = float(self.df.at[idx, "flesch_score"])
        text = str(self.df.at[idx, "testo"])
        sim = self._cosine_similarity(user_profile.topic_vector, doc_embedding)

        if mode == "hybrid":
            breakdown = self.scorer.score(
                semantic_similarity=sim,
                target_readability=user_profile.target_readability,
                doc_readability=doc_readability,
            )
        elif mode == "similarity_only":
            breakdown = self.scorer.score_similarity_only(sim)
        elif mode == "readability_only":
            breakdown = self.scorer.score_readability_only(
                target_readability=user_profile.target_readability,
                doc_readability=doc_readability,
            )
        else:
            raise ValueError(
                f"Unsupported mode '{mode}'. Expected one of: hybrid, similarity_only, readability_only"
            )

        return RecommendationItem(
            doc_id=str(doc_id),
            score=breakdown.score,
            text=text,
            flesch_score=doc_readability,
            semantic_similarity=float(sim),
            normalized_gap=breakdown.normalized_gap,
            penalty=breakdown.penalty,
        )

    def recommender(self, user: Any, doc_id: str) -> Tuple[float, float]:
        """Backward-compatible single-document scoring entrypoint."""

        user_profile = self._coerce_user_profile(user)
        item = self._build_item(user_profile, str(doc_id), mode="hybrid")
        return float(item.score), float(item.flesch_score)

    def rank_top_k_items(self, user: Any, mode: str = "hybrid") -> List[RecommendationItem]:
        """Return top-K scored items with rich metadata."""

        user_profile = self._coerce_user_profile(user)
        candidates = self.catalog(user_profile)

        scored = [
            self._build_item(user_profile, str(doc_id), mode=mode)
            for doc_id in candidates["id"].tolist()
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: self.config.k]

    def rank_top_k(self, user: Any, mode: str = "hybrid") -> Tuple[List[str], np.ndarray, List[str], List[float]]:
        """Return top-K outputs in legacy tuple format used by the app."""

        top_items = self.rank_top_k_items(user, mode=mode)

        titles = [item.doc_id for item in top_items]
        scores = np.round([item.score for item in top_items], 6)
        texts = [item.text for item in top_items]
        flesch_values = [round(item.flesch_score, 2) for item in top_items]

        return titles, scores, texts, flesch_values

    def rank_to_df(self, user: Any, mode: str = "hybrid") -> pd.DataFrame:
        """Return top-K recommendations as a dataframe."""

        top_items = self.rank_top_k_items(user, mode=mode)
        return pd.DataFrame(
            {
                "title": [item.doc_id for item in top_items],
                "score": [item.score for item in top_items],
                "testo": [item.text for item in top_items],
                "flesch_score": [item.flesch_score for item in top_items],
                "semantic_similarity": [item.semantic_similarity for item in top_items],
                "normalized_gap": [item.normalized_gap for item in top_items],
                "penalty": [item.penalty for item in top_items],
            }
        )
    
    
        
        
        
        

        

    

    
    
    
