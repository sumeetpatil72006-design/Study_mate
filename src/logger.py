"""
Logging configuration for the AI Knowledge Assistant.
Includes automatic secret redaction to ensure credentials are never leaked in logs.
"""

import logging
import os
import re
import sys
from typing import Any

class SecretMaskingFilter(logging.Filter):
    """
    Filter that intercepts log records and redacts API keys, credentials, and secrets.
    """
    # Pattern for standard Google API keys: AIzaSy... (39 chars)
    API_KEY_REGEX = re.compile(r"AIza[0-9A-Za-z\-_]{35}")

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._redact(v) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self._redact(arg) if isinstance(arg, str) else arg for arg in record.args)
        return True

    def _redact(self, text: str) -> str:
        # Mask Google API keys pattern
        text = self.API_KEY_REGEX.sub("[REDACTED_KEY]", text)
        # Redact any explicit key currently in environment variables
        for env_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            val = os.getenv(env_var)
            if val and len(val.strip()) > 5 and val.strip() in text:
                text = text.replace(val.strip(), "[REDACTED_KEY]")
        return text

def setup_logger(name: str = "rag_assistant", level: str = None) -> logging.Logger:
    """
    Configures and returns a logger instance with formatted console output
    and automatic secret masking.
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        handler.addFilter(SecretMaskingFilter())
        logger.addHandler(handler)
        logger.addFilter(SecretMaskingFilter())
        logger.propagate = False

    return logger

logger = setup_logger()
