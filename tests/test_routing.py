"""Tests for routing handlers (Chain of Responsibility pattern)."""

import pytest
from unittest.mock import MagicMock

from triage.models.base import ClassificationResult
from triage.routing.handlers import BaseHandler, LocalModelHandler, LLMHandler, HandlerResult


class TestBaseHandler:
    """Tests for BaseHandler base class."""

    def test_set_next_returns_handler(self):
        """Test that set_next returns the next handler."""
        handler1 = BaseHandler()
        handler2 = BaseHandler()
        
        result = handler1.set_next(handler2)
        assert result is handler2

    def test_set_next_chains_handlers(self):
        """Test that handlers can be chained."""
        handler1 = BaseHandler()
        handler2 = BaseHandler()
        handler3 = BaseHandler()
        
        handler1.set_next(handler2).set_next(handler3)
        assert handler1._next is handler2
        assert handler2._next is handler3

    def test_handle_not_implemented(self):
        """Test that _handle is not implemented in base class."""
        handler = BaseHandler()
        with pytest.raises(NotImplementedError):
            handler.handle("test text")


class MockHandler(BaseHandler):
    """Mock handler for testing."""

    def __init__(self, handled: bool = False, label: str = "test", confidence: float = 0.5):
        super().__init__()
        self.handled = handled
        self.label = label
        self.confidence = confidence

    def _handle(self, text: str, labels=None) -> HandlerResult:
        result = ClassificationResult(
            label=self.label,
            confidence=self.confidence,
            comment=f"Handled by {self.__class__.__name__}",
            source="test"
        )
        return HandlerResult(handled=self.handled, result=result)


class TestLocalModelHandler:
    """Tests for LocalModelHandler."""

    def test_initialization(self):
        """Test initializing LocalModelHandler."""
        classifier = MagicMock()
        handler = LocalModelHandler(classifier, threshold=0.80)
        
        assert handler.classifier is classifier
        assert handler.threshold == 0.80

    def test_high_confidence_returns_handled(self):
        """Test that high confidence results return handled=True."""
        classifier = MagicMock()
        classifier.predict.return_value = ClassificationResult(
            label="bug",
            confidence=0.95,
            comment="High confidence",
            source="local"
        )
        
        handler = LocalModelHandler(classifier, threshold=0.80)
        result = handler._handle("test issue")
        
        assert result.handled is True
        assert result.result.label == "bug"
        assert result.result.confidence == 0.95

    def test_low_confidence_returns_unhandled(self):
        """Test that low confidence results return handled=False."""
        classifier = MagicMock()
        classifier.predict.return_value = ClassificationResult(
            label="feature",
            confidence=0.50,
            comment="Low confidence",
            source="local"
        )
        
        handler = LocalModelHandler(classifier, threshold=0.80)
        result = handler._handle("test issue")
        
        assert result.handled is False
        assert result.result.label == "feature"
        assert result.result.confidence == 0.50

    def test_boundary_confidence_exact_threshold(self):
        """Test that exact threshold value is included."""
        classifier = MagicMock()
        classifier.predict.return_value = ClassificationResult(
            label="bug",
            confidence=0.80,
            comment="At threshold",
            source="local"
        )
        
        handler = LocalModelHandler(classifier, threshold=0.80)
        result = handler._handle("test issue")
        
        assert result.handled is True

    def test_metadata_includes_local_info(self):
        """Test that metadata includes local model information."""
        classifier = MagicMock()
        classifier.predict.return_value = ClassificationResult(
            label="critical",
            confidence=0.92,
            comment="",
            source="local"
        )
        
        handler = LocalModelHandler(classifier, threshold=0.80)
        result = handler._handle("test issue")
        
        assert "local" in result.metadata
        assert result.metadata["local"]["label"] == "critical"
        assert result.metadata["local"]["confidence"] == 0.92

    def test_passes_candidate_labels_to_classifier(self):
        """Test that candidate labels are passed to classifier."""
        classifier = MagicMock()
        classifier.predict.return_value = ClassificationResult(
            label="bug", confidence=0.90, comment="", source="local"
        )
        
        handler = LocalModelHandler(classifier, threshold=0.80)
        labels = ["bug", "feature", "docs"]
        handler._handle("test issue", labels=labels)
        
        classifier.predict.assert_called_once_with("test issue", labels)


class TestLLMHandler:
    """Tests for LLMHandler."""

    def test_initialization(self):
        """Test initializing LLMHandler."""
        classifier = MagicMock()
        handler = LLMHandler(classifier)
        
        assert handler.classifier is classifier

    def test_always_returns_handled(self):
        """Test that LLMHandler always returns handled=True."""
        classifier = MagicMock()
        classifier.predict.return_value = ClassificationResult(
            label="feature",
            confidence=0.60,
            comment="LLM result",
            source="llm"
        )
        
        handler = LLMHandler(classifier)
        result = handler._handle("test issue")
        
        assert result.handled is True

    def test_metadata_includes_llm_info(self):
        """Test that metadata includes LLM information."""
        classifier = MagicMock()
        classifier.predict.return_value = ClassificationResult(
            label="documentation",
            confidence=0.75,
            comment="LLM classified this as docs",
            source="llm"
        )
        
        handler = LLMHandler(classifier)
        result = handler._handle("test issue")
        
        assert "llm" in result.metadata
        assert result.metadata["llm"]["label"] == "documentation"
        assert result.metadata["llm"]["confidence"] == 0.75

    def test_passes_candidate_labels_to_classifier(self):
        """Test that candidate labels are passed to classifier."""
        classifier = MagicMock()
        classifier.predict.return_value = ClassificationResult(
            label="bug", confidence=0.80, comment="", source="llm"
        )
        
        handler = LLMHandler(classifier)
        labels = ["bug", "feature"]
        handler._handle("test issue", labels=labels)
        
        classifier.predict.assert_called_once_with("test issue", labels)


class TestChainOfResponsibility:
    """Tests for Chain of Responsibility pattern with multiple handlers."""

    def test_single_handler_high_confidence(self):
        """Test chain with single handler returning handled."""
        classifier = MagicMock()
        classifier.predict.return_value = ClassificationResult(
            label="bug", confidence=0.95, comment="", source="local"
        )
        
        handler = LocalModelHandler(classifier, threshold=0.80)
        result = handler.handle("test issue")
        
        assert result.handled is True
        assert result.result.label == "bug"

    def test_chain_stops_on_first_handled(self):
        """Test that chain stops when first handler returns handled."""
        local_classifier = MagicMock()
        local_classifier.predict.return_value = ClassificationResult(
            label="bug", confidence=0.95, comment="", source="local"
        )
        
        llm_classifier = MagicMock()
        llm_classifier.predict.return_value = ClassificationResult(
            label="feature", confidence=0.80, comment="", source="llm"
        )
        
        local_handler = LocalModelHandler(local_classifier, threshold=0.80)
        llm_handler = LLMHandler(llm_classifier)
        local_handler.set_next(llm_handler)
        
        result = local_handler.handle("test issue")
        
        assert result.handled is True
        assert result.result.label == "bug"
        # LLM should not be called
        llm_classifier.predict.assert_not_called()

    def test_chain_passes_to_next_on_low_confidence(self):
        """Test that chain passes to next handler on low confidence."""
        local_classifier = MagicMock()
        local_classifier.predict.return_value = ClassificationResult(
            label="guess", confidence=0.30, comment="", source="local"
        )
        
        llm_classifier = MagicMock()
        llm_classifier.predict.return_value = ClassificationResult(
            label="bug", confidence=0.90, comment="LLM decision", source="llm"
        )
        
        local_handler = LocalModelHandler(local_classifier, threshold=0.80)
        llm_handler = LLMHandler(llm_classifier)
        local_handler.set_next(llm_handler)
        
        result = local_handler.handle("test issue")
        
        assert result.handled is True
        assert result.result.label == "bug"
        assert result.result.source == "llm"
        llm_classifier.predict.assert_called_once()

    def test_chain_merges_metadata(self):
        """Test that metadata from multiple handlers is merged."""
        local_classifier = MagicMock()
        local_classifier.predict.return_value = ClassificationResult(
            label="maybe", confidence=0.50, comment="", source="local"
        )
        
        llm_classifier = MagicMock()
        llm_classifier.predict.return_value = ClassificationResult(
            label="bug", confidence=0.85, comment="", source="llm"
        )
        
        local_handler = LocalModelHandler(local_classifier, threshold=0.80)
        llm_handler = LLMHandler(llm_classifier)
        local_handler.set_next(llm_handler)
        
        result = local_handler.handle("test issue")
        
        # Both handlers' metadata should be present
        assert "local" in result.metadata
        assert "llm" in result.metadata
        assert result.metadata["local"]["label"] == "maybe"
        assert result.metadata["llm"]["label"] == "bug"

    def test_unhandled_chain_no_next(self):
        """Test that unhandled result is returned when no next handler."""
        local_classifier = MagicMock()
        local_classifier.predict.return_value = ClassificationResult(
            label="uncertain", confidence=0.40, comment="", source="local"
        )
        
        local_handler = LocalModelHandler(local_classifier, threshold=0.80)
        result = local_handler.handle("test issue")
        
        assert result.handled is False
        assert result.result.label == "uncertain"

    def test_candidate_labels_passed_through_chain(self):
        """Test that candidate labels are passed through the entire chain."""
        local_classifier = MagicMock()
        llm_classifier = MagicMock()
        
        local_classifier.predict.return_value = ClassificationResult(
            label="maybe", confidence=0.50, comment="", source="local"
        )
        llm_classifier.predict.return_value = ClassificationResult(
            label="bug", confidence=0.90, comment="", source="llm"
        )
        
        local_handler = LocalModelHandler(local_classifier, threshold=0.80)
        llm_handler = LLMHandler(llm_classifier)
        local_handler.set_next(llm_handler)
        
        labels = ["bug", "feature", "docs"]
        result = local_handler.handle("test issue", labels=labels)
        
        # Both should be called with labels
        local_classifier.predict.assert_called_once_with("test issue", labels)
        llm_classifier.predict.assert_called_once_with("test issue", labels)
