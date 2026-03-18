"""Configuration loading for the triage system.

This module loads environment variables, optionally via python-dotenv.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


@dataclass(frozen=True)
class Config:
    """Runtime configuration values.

    Args:
        github_token: GitHub Personal Access Token.
        openrouter_api_key: OpenRouter API key.
        openrouter_model: OpenRouter model name.
        openrouter_timeout: OpenRouter request timeout seconds.
        github_api_url: GitHub API base URL.
        confidence_threshold: Local model threshold.
        model_path: Path to the serialized model file.
        log_path: Path to the telemetry log file.
        fallback_label: Label to use when no label matches.
        dry_run: Whether to avoid mutating GitHub.
    """

    github_token: str
    openrouter_api_key: str
    openrouter_model: str
    openrouter_timeout: int
    github_api_url: str
    confidence_threshold: float
    model_path: Path
    log_path: Path
    fallback_label: str
    dry_run: bool


DEFAULT_GITHUB_API_URL = "https://api.github.com"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPENROUTER_TIMEOUT = 45
DEFAULT_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_MODEL_PATH = Path("models/issue_classifier.joblib")
DEFAULT_LOG_PATH = Path("logs/triage.jsonl")
DEFAULT_FALLBACK_LABEL = "needs-triage"


def _get_bool(value: Optional[str], default: bool = False) -> bool:
    """Parse a boolean environment value.

    Args:
        value: String value from environment.
        default: Default value if missing.

    Returns:
        Parsed boolean.
    """

    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config() -> Config:
    """Load configuration from environment variables.

    Returns:
        Config object with resolved values.
    """

    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    return Config(
        github_token=github_token,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip(),
        openrouter_timeout=int(
            os.getenv("OPENROUTER_TIMEOUT", str(DEFAULT_OPENROUTER_TIMEOUT))
        ),
        github_api_url=os.getenv("GITHUB_API_URL", DEFAULT_GITHUB_API_URL).strip(),
        confidence_threshold=float(
            os.getenv("CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD))
        ),
        model_path=Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser(),
        log_path=Path(os.getenv("LOG_PATH", str(DEFAULT_LOG_PATH))).expanduser(),
        fallback_label=os.getenv("FALLBACK_LABEL", DEFAULT_FALLBACK_LABEL).strip(),
        dry_run=_get_bool(os.getenv("DRY_RUN"), default=False),
    )
