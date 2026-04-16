from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.recommender.models import READABILITY_MAX, READABILITY_MIN
from utils.data_loader import load_embedding, load_features_df
from utils.io_utils import load_json, load_yaml, save_json


@dataclass(frozen=True)
class UserModelConfig:
    """Configuration for user profile persistence and adaptation."""

    users_path: Path
    init_vector_path: Path
    gamma: float = 0.6
    readability_min: float = READABILITY_MIN
    readability_max: float = READABILITY_MAX
    min_shift_map: Dict[int, float] = field(
        default_factory=lambda: {
            1: 0.25,
            2: 0.5,
            3: 0.0,
            4: 0.5,
            5: 0.75,
        }
    )

    @classmethod
    def from_project_config(cls, project_root: str) -> "UserModelConfig":
        """Build config from the repository YAML file."""

        project_path = Path(project_root)
        config = load_yaml()
        rel_users = config["paths"]["user_json"]
        gamma = float(config.get("gamma", 0.6))
        return cls(
            users_path=project_path / rel_users,
            init_vector_path=project_path / "topic_vector_init.npy",
            gamma=gamma,
        )


class UserModelService:
    """Service class that manages user profile lifecycle and updates."""

    def __init__(self, config: UserModelConfig) -> None:
        self.config = config
        self.config.users_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_vector(vector: Iterable[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=float).reshape(-1)
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector")
        return arr / norm

    def initialize_topic_vector(self, embedding: Optional[Iterable[Iterable[float]]] = None) -> np.ndarray:
        """Create the initial topic vector as normalized centroid of normalized embeddings."""

        emb = np.asarray(load_embedding() if embedding is None else embedding, dtype=float)
        if emb.ndim != 2:
            raise ValueError("Embedding must be a 2D matrix")

        row_norms = np.linalg.norm(emb, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        normalized_rows = emb / row_norms

        centroid = normalized_rows.mean(axis=0)
        centroid = self._normalize_vector(centroid)

        np.save(self.config.init_vector_path, centroid)
        return centroid

    def _get_default_topic_vector(self) -> np.ndarray:
        """Load cached default topic vector or regenerate it if missing."""

        if not self.config.init_vector_path.exists():
            return self.initialize_topic_vector()

        vector = np.load(self.config.init_vector_path)
        return self._normalize_vector(vector)

    def save_user_json(self, user: Mapping[str, object], user_id: Optional[int] = None) -> None:
        """Persist a user profile to disk."""

        persisted = dict(user)
        resolved_user_id = user_id if user_id is not None else int(persisted["user_id"])
        persisted["user_id"] = resolved_user_id

        target = float(persisted.get("target_readability", 60.0))
        persisted["target_readability"] = float(
            np.clip(target, self.config.readability_min, self.config.readability_max)
        )

        history = persisted.get("history", [])
        persisted["history"] = [str(doc_id) for doc_id in history]

        output_path = self.config.users_path / f"user{resolved_user_id}.json"
        save_json(persisted, str(output_path))

    def build_user_model(
        self,
        user_id: int,
        *,
        topic_vector_default: Optional[Iterable[float]] = None,
        default_readability: float = 60,
        save: bool = True,
    ) -> Dict[str, object]:
        """Create and optionally persist a new user profile."""

        topic_vector = (
            self._get_default_topic_vector()
            if topic_vector_default is None
            else self._normalize_vector(topic_vector_default)
        )

        profile = {
            "user_id": int(user_id),
            "topic_vector": topic_vector.tolist(),
            "target_readability": float(
                np.clip(default_readability, self.config.readability_min, self.config.readability_max)
            ),
            "history": [],
        }

        if save:
            self.save_user_json(profile, int(user_id))
        return profile

    def load_user_model(self, name: str, path: Optional[str] = None) -> Dict[str, object]:
        """Load a user profile from JSON."""

        users_dir = Path(path) if path is not None else self.config.users_path
        path_user = users_dir / name

        if not path_user.exists():
            raise FileNotFoundError(f"User profile not found: {path_user}")

        user_json = load_json(str(path_user))
        return {
            "user_id": user_json["user_id"],
            "topic_vector": user_json["topic_vector"],
            "target_readability": float(user_json["target_readability"]),
            "history": [str(doc_id) for doc_id in user_json.get("history", [])],
        }

    @staticmethod
    def difficulty_to_alpha(difficulty: int) -> float:
        """Map feedback q in [1, 5] to alpha_q = 0.1 * q."""

        clipped = int(np.clip(difficulty, 1, 5))
        return 0.1 * clipped

    def update_topic_vector(
        self,
        user: Mapping[str, object],
        doc_embedding: Iterable[float],
        difficulty: int,
    ) -> np.ndarray:
        """Update semantic user vector with interpolation rule from the paper."""

        alpha_q = self.difficulty_to_alpha(difficulty)
        old_vector = self._normalize_vector(user["topic_vector"])
        new_embedding = self._normalize_vector(doc_embedding)

        updated_vector = (1 - alpha_q) * old_vector + alpha_q * new_embedding
        return self._normalize_vector(updated_vector)

    def update_target_readability(
        self,
        old_target: float,
        doc_readability: float,
        difficulty: int,
        *,
        gamma: Optional[float] = None,
        min_shift_map: Optional[Mapping[int, float]] = None,
    ) -> float:
        """Update readability target using Delta tau = gamma * f(q) * (r_d - tau).

        A small minimum shift per feedback category can be applied to avoid vanishing
        updates when user feedback is informative.
        """

        old_target_clipped = float(
            np.clip(old_target, self.config.readability_min, self.config.readability_max)
        )
        doc_readability_clipped = float(np.clip(doc_readability, 0.0, 100.0))

        gamma_value = self.config.gamma if gamma is None else float(gamma)
        alpha_q = self.difficulty_to_alpha(difficulty)
        delta = gamma_value * alpha_q * (doc_readability_clipped - old_target_clipped)

        thresholds = self.config.min_shift_map if min_shift_map is None else min_shift_map
        minimum_shift = float(thresholds.get(int(np.clip(difficulty, 1, 5)), 0.0))
        if delta != 0 and abs(delta) < minimum_shift:
            delta = float(np.sign(delta) * minimum_shift)

        new_target = old_target_clipped + delta
        return float(np.clip(new_target, self.config.readability_min, self.config.readability_max))

    @staticmethod
    def update_history(user: Dict[str, object], doc_id: object) -> None:
        """Append document to history if not already present."""

        doc_key = str(doc_id)
        history = [str(item) for item in user.get("history", [])]
        if doc_key not in history:
            history.append(doc_key)
        user["history"] = history

    @staticmethod
    def get_document_embedding(doc_id: object) -> np.ndarray:
        """Fetch document embedding by document ID using cached loaders."""

        doc_key = str(doc_id)
        df = load_features_df().copy()
        df["id"] = df["id"].astype(str)

        idx_array = df.index[df["id"] == doc_key].tolist()
        if not idx_array:
            raise ValueError(f"Document not found: {doc_key}")

        emb = np.asarray(load_embedding(), dtype=float)
        return emb[idx_array[0]].reshape(-1)

    def update_user_model(
        self,
        user: Dict[str, object],
        doc_id: object,
        doc_readability: float,
        difficulty: int,
    ) -> Dict[str, object]:
        """Update full user state after reading feedback."""

        self.update_history(user, doc_id)
        doc_embedding = self.get_document_embedding(doc_id)

        new_vector = self.update_topic_vector(user, doc_embedding, difficulty)
        new_target = self.update_target_readability(
            old_target=float(user["target_readability"]),
            doc_readability=float(doc_readability),
            difficulty=int(difficulty),
        )

        user["topic_vector"] = new_vector.tolist()
        user["target_readability"] = float(new_target)
        self.save_user_json(user, int(user["user_id"]))

        return user


_DEFAULT_SERVICE = UserModelService(UserModelConfig.from_project_config(PROJECT_ROOT))


def save_user_json(user, user_id):
    """Backward-compatible wrapper for profile persistence."""

    _DEFAULT_SERVICE.save_user_json(user, int(user_id))


def initialize_topic_vector(embedding):
    """Backward-compatible wrapper for initial topic vector generation."""

    return _DEFAULT_SERVICE.initialize_topic_vector(embedding)


def build_user_model(user_id, *, topic_vector_default=None, default_readability=60, save=True):
    """Backward-compatible wrapper for user profile creation."""

    return _DEFAULT_SERVICE.build_user_model(
        int(user_id),
        topic_vector_default=topic_vector_default,
        default_readability=default_readability,
        save=save,
    )


def load_user_model(name, path):
    """Backward-compatible wrapper for user profile loading."""

    return _DEFAULT_SERVICE.load_user_model(name, path)


def difficulty_to_alpha(difficulty):
    """Backward-compatible wrapper for feedback-to-alpha mapping."""

    return _DEFAULT_SERVICE.difficulty_to_alpha(int(difficulty))


def update_topic_vector(user, doc_embedding, difficulty):
    """Backward-compatible wrapper for topic vector updates."""

    return _DEFAULT_SERVICE.update_topic_vector(user, doc_embedding, int(difficulty))


def update_target_readability(
    old_target,
    doc_readability,
    difficulty,
    learning_rate=1.0,
    damping=0.6,
):
    """Backward-compatible wrapper for readability updates.

    The implementation follows the paper formula. `learning_rate` is retained only
    for call-site compatibility and is not used.
    """

    _ = learning_rate
    return _DEFAULT_SERVICE.update_target_readability(
        old_target=old_target,
        doc_readability=doc_readability,
        difficulty=int(difficulty),
        gamma=float(damping),
    )


def update_history(user, doc_id):
    """Backward-compatible wrapper for history updates."""

    _DEFAULT_SERVICE.update_history(user, doc_id)


def update_user_model(user, doc_id, doc_readability, difficulty):
    """Backward-compatible wrapper for full user model updates."""

    return _DEFAULT_SERVICE.update_user_model(user, doc_id, doc_readability, int(difficulty))














    
    



