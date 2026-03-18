"""Shared pytest fixtures and configuration."""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import pytest

from triage.models.base import ClassificationResult


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_model_path(temp_dir):
    """Create a temporary model path."""
    return temp_dir / "test_model.joblib"


@pytest.fixture
def temp_log_path(temp_dir):
    """Create a temporary log file path."""
    return temp_dir / "test_log.jsonl"


@pytest.fixture
def sample_issue_text():
    """Sample issue text for testing."""
    return """# Bug: App crashes on startup

## Description
The application crashes immediately after launching on version 1.2.3.

## Steps to Reproduce
1. Install version 1.2.3
2. Launch the app
3. Observe crash

## Expected Behavior
App should launch without errors.

## Actual Behavior
App crashes with segfault.

## Environment
- OS: Ubuntu 22.04
- Python: 3.10.5
"""


@pytest.fixture
def sample_github_issue_response():
    """Sample GitHub API issue response."""
    return {
        "id": 12345,
        "number": 42,
        "title": "Bug: App crashes",
        "body": "The app crashes on startup.",
        "state": "open",
        "url": "https://github.com/owner/repo/issues/42",
        "html_url": "https://github.com/owner/repo/issues/42",
        "labels": [
            {"name": "bug"},
            {"name": "high-priority"}
        ],
        "user": {"login": "testuser"},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_openrouter_response():
    """Sample OpenRouter API response."""
    return {
        "id": "chat-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"label": "bug", "confidence": 0.92, "comment": "Classified as bug based on crash report."}'
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200
        }
    }


@pytest.fixture
def sample_classification_result():
    """Sample classification result."""
    return ClassificationResult(
        label="bug",
        confidence=0.85,
        comment="This is a bug report.",
        source="local"
    )


@pytest.fixture
def mock_github_adapter():
    """Mock GitHub adapter."""
    adapter = MagicMock()
    adapter.iter_issues.return_value = []
    adapter.list_labels.return_value = ["bug", "feature", "documentation"]
    adapter.add_labels.return_value = None
    adapter.create_comment.return_value = None
    return adapter


@pytest.fixture
def mock_openrouter_adapter():
    """Mock OpenRouter adapter."""
    adapter = MagicMock()
    adapter.classify_issue.return_value = ClassificationResult(
        label="bug",
        confidence=0.85,
        comment="Classified by LLM",
        source="llm"
    )
    return adapter


@pytest.fixture
def sample_training_data():
    """Sample training texts and labels."""
    texts = [
        "The app crashes on startup with a segfault error.",
        "Add support for dark mode theme.",
        "Update the API documentation with new endpoints.",
        "Memory leak in the connection pool.",
        "Please add support for OAuth2 authentication.",
        "Fix typos in README.md.",
    ]
    labels = ["bug", "feature", "documentation", "bug", "feature", "documentation"]
    return texts, labels


@pytest.fixture
def mock_sklearn_pipeline(temp_model_path):
    """Create a mock sklearn pipeline and save it."""
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    vectorizer = TfidfVectorizer(max_features=100)
    clf = LogisticRegression(max_iter=100)
    pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    
    # Fit on dummy data with multiple classes (required by LogisticRegression)
    pipeline.fit(["bug report", "feature request"], ["bug", "feature"])
    
    # Save
    temp_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, temp_model_path)
    
    return temp_model_path


@pytest.fixture
def config_dict():
    """Sample configuration dictionary."""
    return {
        "github_token": "ghp_test_token",
        "openrouter_api_key": "sk_test_key",
        "openrouter_model": "openai/gpt-4o-mini",
        "openrouter_timeout": 45,
        "github_api_url": "https://api.github.com",
        "confidence_threshold": 0.80,
        "model_path": Path("models/issue_classifier.joblib"),
        "log_path": Path("logs/triage.jsonl"),
        "fallback_label": "needs-triage",
        "dry_run": False,
    }
