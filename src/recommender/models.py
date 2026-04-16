from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set

import numpy as np

READABILITY_MIN = 20.0
READABILITY_MAX = 90.0


@dataclass(frozen=True)
class RecommenderConfig:
    """Typed recommender hyperparameters used across scoring and ranking."""

    k: int = 10
    eta: float = 0.6
    zeta: float = 0.4
    alpha: float = 0.4
    tol: float = 15.0

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]]) -> "RecommenderConfig":
        """Build a typed config from a plain dict, falling back to defaults."""

        source = config or {}
        return cls(
            k=int(source.get("k", cls.k)),
            eta=float(source.get("eta", cls.eta)),
            zeta=float(source.get("zeta", cls.zeta)),
            alpha=float(source.get("alpha", cls.alpha)),
            tol=float(source.get("tol", cls.tol)),
        )

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be > 0")
        if self.tol <= 0:
            raise ValueError("tol must be > 0")


@dataclass
class UserProfile:
    """Runtime representation of a user profile."""

    user_id: str
    topic_vector: np.ndarray
    target_readability: float
    history: Set[str] = field(default_factory=set)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Create a UserProfile from the persisted JSON-compatible dictionary."""

        return cls(
            user_id=str(data.get("user_id", "unknown")),
            topic_vector=np.asarray(data.get("topic_vector", []), dtype=float),
            target_readability=float(data.get("target_readability", 60.0)),
            history={str(item) for item in data.get("history", [])},
        )

    def __post_init__(self) -> None:
        self.topic_vector = np.asarray(self.topic_vector, dtype=float).reshape(-1)
        self.target_readability = float(
            np.clip(self.target_readability, READABILITY_MIN, READABILITY_MAX)
        )
        self.history = {str(item) for item in self.history}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to a JSON-compatible dictionary."""

        return {
            "user_id": int(self.user_id) if str(self.user_id).isdigit() else self.user_id,
            "topic_vector": self.topic_vector.astype(float).tolist(),
            "target_readability": float(self.target_readability),
            "history": sorted(self.history),
        }


@dataclass(frozen=True)
class RecommendationItem:
    """Scored candidate returned by the ranking pipeline."""

    doc_id: str
    score: float
    text: str
    flesch_score: float
    semantic_similarity: float
    normalized_gap: float
    penalty: float


@dataclass(frozen=True)
class ScoringBreakdown:
    """Detailed scoring components for inspection and testing."""

    score: float
    semantic_similarity: float
    normalized_gap: float
    penalty: float
