from pathlib import Path

import pytest

from src.user.model_user import UserModelConfig, UserModelService


def _build_service(tmp_path: Path, min_shift_map=None) -> UserModelService:
    config = UserModelConfig(
        users_path=tmp_path / "users",
        init_vector_path=tmp_path / "topic_vector_init.npy",
        gamma=0.6,
        min_shift_map=min_shift_map
        or {
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
            5: 0.0,
        },
    )
    return UserModelService(config)


def test_build_and_load_user_model(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.initialize_topic_vector([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    user = service.build_user_model(user_id=7, default_readability=65, save=True)
    loaded = service.load_user_model("user7.json", str(service.config.users_path))

    assert user["user_id"] == 7
    assert loaded["user_id"] == 7
    assert loaded["target_readability"] == pytest.approx(65.0)
    assert len(loaded["topic_vector"]) == 3


def test_target_readability_update_follows_paper_formula(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    updated = service.update_target_readability(
        old_target=60,
        doc_readability=70,
        difficulty=5,
    )
    # Delta tau = gamma * f(q) * (r_d - tau) = 0.6 * 0.5 * 10 = 3
    assert updated == pytest.approx(63.0)


def test_target_update_applies_minimum_shift_when_enabled(tmp_path: Path) -> None:
    service = _build_service(
        tmp_path,
        min_shift_map={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1.0},
    )

    updated = service.update_target_readability(
        old_target=60,
        doc_readability=60.1,
        difficulty=5,
    )
    # Raw delta would be 0.6 * 0.5 * 0.1 = 0.03, then raised to min shift 1.0.
    assert updated == pytest.approx(61.0)
