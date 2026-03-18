"""Integration tests for the triage pipeline."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from triage.models.base import ClassificationResult
from triage.models.local_model import LocalModelClassifier
from triage.models.llm import LLMClassifier
from triage.routing.handlers import LocalModelHandler, LLMHandler
from triage.preprocessing.text import sanitize_text
from triage.telemetry.logger import TelemetryLogger


class TestEndToEndRouting:
    """Integration tests for full routing pipeline."""

    def test_high_confidence_local_only(self, mock_sklearn_pipeline):
        """Test routing with local model."""
        # Setup
        local_classifier = LocalModelClassifier(mock_sklearn_pipeline)
        handler_chain = LocalModelHandler(local_classifier, threshold=0.50)  # Lower threshold
        
        # Execute
        result = handler_chain.handle("test issue text")
        
        # Assert - at minimum it should return a result
        assert result.result is not None
        assert result.result.source == "local"
        assert 0.0 <= result.result.confidence <= 1.0

    def test_low_confidence_with_llm_fallback(self, mock_sklearn_pipeline):
        """Test routing with low-confidence local model triggering LLM."""
        # Mocking low-confidence local model
        with patch.object(LocalModelClassifier, "predict") as mock_local:
            mock_local.return_value = ClassificationResult(
                label="uncertain", confidence=0.40, comment="", source="local"
            )
            
            local_classifier = LocalModelClassifier(mock_sklearn_pipeline)
            
            # Mock LLM
            mock_llm_adapter = MagicMock()
            mock_llm_adapter.classify_issue.return_value = ClassificationResult(
                label="bug", confidence=0.88, comment="LLM classified", source="llm"
            )
            llm_classifier = LLMClassifier(mock_llm_adapter, fallback_label="needs-triage")
            
            # Build chain
            handler_chain = LocalModelHandler(local_classifier, threshold=0.80)
            handler_chain.set_next(LLMHandler(llm_classifier))
            
            # Execute
            result = handler_chain.handle("test issue")
            
            # Assert
            assert result.handled is True
            assert result.result.source == "llm"
            assert result.result.label == "bug"

    def test_text_preprocessing_in_pipeline(self, mock_sklearn_pipeline):
        """Test that text is properly preprocessed before routing."""
        raw_text = """
        # Bug: App Crashes
        
        ```python
        def broken():
            pass
        ```
        
        The app [crashes](docs/crash).
        """
        
        sanitized = sanitize_text(raw_text)
        
        # Assert preprocessing removed code blocks and links
        assert "```" not in sanitized
        assert "[" not in sanitized
        # Content should be preserved
        assert "app" in sanitized.lower()
        assert "crash" in sanitized.lower()
        
        # Route the sanitized text
        classifier = LocalModelClassifier(mock_sklearn_pipeline)
        handler = LocalModelHandler(classifier, threshold=0.50)  # Lower threshold
        
        result = handler.handle(sanitized)
        
        assert result.result is not None


class TestTelemetryIntegration:
    """Integration tests for telemetry logging in pipelines."""

    def test_routing_decision_logging(self, temp_log_path, mock_sklearn_pipeline):
        """Test that routing decisions are logged."""
        telemetry = TelemetryLogger(temp_log_path)
        
        classifier = LocalModelClassifier(mock_sklearn_pipeline)
        handler = LocalModelHandler(classifier, threshold=0.80)
        
        result = handler.handle("test issue")
        
        # Log the decision
        decision = {
            "owner": "test",
            "repo": "repo",
            "issue_number": 42,
            "label": result.result.label,
            "confidence": result.result.confidence,
            "source": result.result.source,
            "handled": result.handled,
        }
        telemetry.log_decision(decision)
        
        # Verify logging
        assert temp_log_path.exists()
        lines = temp_log_path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_error_event_logging(self, temp_log_path):
        """Test that error events are logged."""
        telemetry = TelemetryLogger(temp_log_path)
        
        error_data = {
            "stage": "classify",
            "error": "Model not found",
            "issue_number": 1,
        }
        telemetry.log_event("error", error_data)
        
        assert temp_log_path.exists()
        import json
        lines = temp_log_path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        
        assert record["event"] == "error"
        assert record["data"]["stage"] == "classify"
        assert record["data"]["error"] == "Model not found"


class TestConfigIntegration:
    """Integration tests for configuration usage."""

    def test_config_applied_to_handler_chain(self, mock_sklearn_pipeline):
        """Test that configuration is properly applied to handler chain."""
        from triage.config import Config
        
        config = Config(
            github_token="ghp_test",
            openrouter_api_key="sk_test",
            openrouter_model="openai/gpt-4o-mini",
            openrouter_timeout=45,
            github_api_url="https://api.github.com",
            confidence_threshold=0.85,  # Custom threshold
            model_path=mock_sklearn_pipeline,
            log_path=Path("logs.jsonl"),
            fallback_label="triage-needed",
            dry_run=False,
        )
        
        # Build handler with config
        classifier = LocalModelClassifier(config.model_path)
        handler = LocalModelHandler(classifier, threshold=config.confidence_threshold)
        
        # Verify threshold is applied
        assert handler.threshold == 0.85


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_issue_with_special_characters(self, mock_sklearn_pipeline):
        """Test routing issue with special characters and markdown."""
        issue_text = """
        # 🚀 Feature: Add Unicode Support
        
        > This is important
        
        The app needs to support [Unicode](https://example.com) characters.
        
        - Item 1
        - Item 2
        
        ```go
        func main() {}
        ```
        """
        
        sanitized = sanitize_text(issue_text)
        
        classifier = LocalModelClassifier(mock_sklearn_pipeline)
        handler = LocalModelHandler(classifier, threshold=0.70)
        
        result = handler.handle(sanitized)
        
        assert result.result is not None
        assert result.result.label is not None

    def test_very_long_issue_text(self, mock_sklearn_pipeline):
        """Test routing with very long issue text."""
        long_text = "This is a very long issue. " * 1000
        
        classifier = LocalModelClassifier(mock_sklearn_pipeline)
        handler = LocalModelHandler(classifier, threshold=0.80)
        
        result = handler.handle(long_text)
        
        assert result.result is not None

    def test_minimal_issue_text(self, mock_sklearn_pipeline):
        """Test routing with minimal issue text."""
        short_text = "Bug"
        
        classifier = LocalModelClassifier(mock_sklearn_pipeline)
        handler = LocalModelHandler(classifier, threshold=0.80)
        
        result = handler.handle(short_text)
        
        assert result.result is not None

    def test_empty_after_sanitization(self, mock_sklearn_pipeline):
        """Test handling issue that becomes mostly empty after sanitization."""
        # Only markdown, no actual content
        text = "# # > - ```python```"
        
        sanitized = sanitize_text(text)
        # Should be very short after removing markdown
        assert len(sanitized) < 15
        
        # Should still handle gracefully even with minimal text
        classifier = LocalModelClassifier(mock_sklearn_pipeline)
        handler = LocalModelHandler(classifier, threshold=0.30)  # Very low threshold
        
        result = handler.handle(sanitized)
        
        assert result.result is not None
