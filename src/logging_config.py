"""Logging setup for the RAG demo.

Call ``configure_logging`` once at application start. Other modules
should simply obtain loggers via ``logging.getLogger(__name__)``.
"""

import logging
import os
from typing import Optional


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logger with sane defaults.

    The first call sets up logging; subsequent calls are no-ops because
    ``basicConfig`` ignores repeated invocations once handlers exist.
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
