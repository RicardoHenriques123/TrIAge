"""Tests for classifier models."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from triage.models.base import ClassificationResult
from triage.models.local_model import LocalModelClassifier
from triage.models.llm import LLMClassifier


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_creation(self):
        """Test creating a ClassificationResult."""
        result = ClassificationResult(
            label="bug",
            confidence=0.95,
            comment="Test comment",
            source="local"
        )
        
        assert result.label == "bug"
        assert result.confidence == 0.95
        assert result.comment == "Test comment"
        assert result.source == "local"

    def test_default_values(self):
        """Test default values for optional fields."""
        result = ClassificationResult(label="feature", confidence=0.75)
        
        assert result.label == "feature"
        assert result.confidence == 0.75
        assert result.comment == ""
        assert result.source == ""


class TestLocalModelClassifier:
    """Tests for LocalModelClassifier."""

    def test_missing_model_file(self):
        """Test that missing model file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            LocalModelClassifier(Path("/nonexistent/model.joblib"))

    def test_initialization_with_valid_model(self, mock_sklearn_pipeline):
        """Test initializing with a valid model file."""
        classifier = LocalModelClassifier(mock_sklearn_pipeline)
        assert classifier.model_path == mock_sklearn_pipeline
        assert classifier.pipeline is not None

    def test_predict_with_valid_model(self, mock_sklearn_pipeline):
        """Test prediction with a valid model."""
        classifier = LocalModelClassifier(mock_sklearn_pipeline)
        result = classifier.predict("test issue")
        
        assert isinstance(result, ClassificationResult)
        assert result.label is not None
        assert 0.0 <= result.confidence <= 1.0
        assert result.source == "local"

    def test_predict_with_candidate_labels(self, mock_sklearn_pipeline):
        """Test prediction with candidate labels."""
        classifier = LocalModelClassifier(mock_sklearn_pipeline)
        result = classifier.predict("test issue", labels=["bug", "feature"])
        
        assert isinstance(result, ClassificationResult)
        assert result.label is not None

    def test_corrupted_model_file(self, temp_model_path):
        """Test that corrupted model file raises an error."""
        # Write garbage to the file
        temp_model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_model_path.write_text("this is not a valid joblib file")
        
        # Should raise an error when loading
        with pytest.raises((RuntimeError, ValueError, IndexError, OSError)):
            LocalModelClassifier(temp_model_path)


class TestLLMClassifier:
    """Tests for LLMClassifier."""

    def test_initialization(self, mock_openrouter_adapter):
        """Test initializing LLMClassifier."""
        classifier = LLMClassifier(
            adapter=mock_openrouter_adapter,
            fallback_label="needs-triage"
        )
        assert classifier.adapter == mock_openrouter_adapter
        assert classifier.fallback_label == "needs-triage"

    def test_predict_valid_response(self, mock_openrouter_adapter):
        """Test prediction with valid adapter response."""
        mock_openrouter_adapter.classify_issue.return_value = ClassificationResult(
            label="bug",
            confidence=0.92,
            comment="This is a bug",
            source="llm"
        )
        
        classifier = LLMClassifier(mock_openrouter_adapter, fallback_label="triage")
        result = classifier.predict("issue text")
        
        assert result.label == "bug"
        assert result.confidence == 0.92
        assert result.source == "llm"

    def test_predict_with_candidate_labels(self, mock_openrouter_adapter):
        """Test prediction passes candidate labels to adapter."""
        classifier = LLMClassifier(mock_openrouter_adapter, fallback_label="triage")
        labels = ["bug", "feature", "docs"]
        
        classifier.predict("issue text", labels=labels)
        
        mock_openrouter_adapter.classify_issue.assert_called_once_with(
            "issue text", labels
        )

    def test_predict_uses_fallback_when_label_not_in_candidates(self, mock_openrouter_adapter):
        """Test that fallback label is used when predicted label not in candidates."""
        mock_openrouter_adapter.classify_issue.return_value = ClassificationResult(
            label="invalid-label",
            confidence=0.85,
            comment="",
            source="llm"
        )
        
        classifier = LLMClassifier(mock_openrouter_adapter, fallback_label="triage")
        result = classifier.predict("issue text", labels=["bug", "feature"])
        
        assert result.label == "triage"

    def test_predict_clamps_confidence(self, mock_openrouter_adapter):
        """Test that confidence is clamped to [0, 1]."""
        mock_openrouter_adapter.classify_issue.return_value = ClassificationResult(
            label="bug",
            confidence=1.5,  # Invalid value
            comment="",
            source="llm"
        )
        
        classifier = LLMClassifier(mock_openrouter_adapter, fallback_label="triage")
        result = classifier.predict("issue text")
        
        assert result.confidence == 1.0

    def test_predict_negative_confidence_clamped(self, mock_openrouter_adapter):
        """Test that negative confidence is clamped to 0."""
        mock_openrouter_adapter.classify_issue.return_value = ClassificationResult(
            label="bug",
            confidence=-0.5,
            comment="",
            source="llm"
        )
        
        classifier = LLMClassifier(mock_openrouter_adapter, fallback_label="triage")
        result = classifier.predict("issue text")
        
        assert result.confidence == 0.0

    def test_predict_error_detection(self, mock_openrouter_adapter):
        """Test that LLM errors are detected from comment."""
        mock_openrouter_adapter.classify_issue.return_value = ClassificationResult(
            label="bug",
            confidence=0.0,
            comment="LLM request failed: connection timeout",
            source="llm"
        )
        
        classifier = LLMClassifier(mock_openrouter_adapter, fallback_label="triage")
        result = classifier.predict("issue text")
        
        assert result.source == "llm_error"

    def test_predict_openrouter_error_detection(self, mock_openrouter_adapter):
        """Test detection of OpenRouter-specific errors."""
        mock_openrouter_adapter.classify_issue.return_value = ClassificationResult(
            label="bug",
            confidence=0.0,
            comment="OpenRouter request failed: rate limit exceeded",
            source="llm"
        )
        
        classifier = LLMClassifier(mock_openrouter_adapter, fallback_label="triage")
        result = classifier.predict("issue text")
        
        assert result.source == "llm_error"

    def test_predict_parsing_error_detection(self, mock_openrouter_adapter):
        """Test detection of parsing errors."""
        mock_openrouter_adapter.classify_issue.return_value = ClassificationResult(
            label="bug",
            confidence=0.0,
            comment="OpenRouter response parsing failed: invalid JSON",
            source="llm"
        )
        
        classifier = LLMClassifier(mock_openrouter_adapter, fallback_label="triage")
        result = classifier.predict("issue text")
        
        assert result.source == "llm_error"
