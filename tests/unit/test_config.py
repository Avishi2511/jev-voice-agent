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


def test_log_level_filters_debug(capsys: pytest.CaptureFixture[str]) -> None:
    import structlog

    from jevagent.observability.logging import configure_logging

    try:
        configure_logging("INFO")
        log = structlog.get_logger("t")
        log.debug("quiet_event")
        log.info("loud_event")
        out = capsys.readouterr().out
        assert "quiet_event" not in out and "loud_event" in out
        configure_logging("debug")
        structlog.get_logger("t").debug("now_visible")
        assert "now_visible" in capsys.readouterr().out
        with pytest.raises(ValueError, match="unknown log level"):
            configure_logging("chatty")
    finally:
        structlog.reset_defaults()
