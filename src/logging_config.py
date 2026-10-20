"""Logging configuration."""

import logging
import logging.handlers
import sys
from pathlib import Path

from src.config import get_settings


def setup_logging(settings=None) -> logging.Logger:
    """Configure application logging."""
    if settings is None:
        settings = get_settings()

    log_config = settings.logging
    log_file = settings.paths.logs / log_config.file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_config.level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(log_config.format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_config.level.upper()))
    root_logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_config.max_bytes,
        backupCount=log_config.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_config.level.upper()))
    root_logger.addHandler(file_handler)

    # Reduce noise from third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)