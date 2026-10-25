"""Process-wide log level for structlog (default INFO, so debug stream events stay quiet)."""

import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    levels = logging.getLevelNamesMapping()
    name = level.upper()
    if name not in levels:
        raise ValueError(f"unknown log level {level!r}; use one of {sorted(levels)}")
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(levels[name]))
