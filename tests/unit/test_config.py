import pytest
from pydantic import ValidationError

from jevagent.config import Settings


def test_default_is_pinned() -> None:
    assert Settings(_env_file=None).jev_model == "jev-1.14.0"  # type: ignore[call-arg]


def test_latest_allowed_in_dev() -> None:
    s = Settings(_env_file=None, app_env="dev", jev_model="jev-latest")  # type: ignore[call-arg]
    assert s.jev_model == "jev-latest"


@pytest.mark.parametrize("env", ["staging", "prod"])
def test_latest_rejected_outside_dev(env: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env=env, jev_model="jev-latest")  # type: ignore[call-arg]


def test_example_env_file_loads() -> None:
    """The shipped .env.example must be usable as is once keys are filled in."""
    from pathlib import Path

    s = Settings(_env_file=Path(__file__).parents[2] / ".env.example")  # type: ignore[call-arg]
    assert s.jev_model == "jev-1.14.0"
    assert s.ab_hybrid_percent is None and s.baseline_mode is False and s.shadow_mode is False
    assert s.public_url == "" and s.save_transcripts is False
