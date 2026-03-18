"""Chain-of-responsibility routing handlers."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from triage.models.base import ClassificationResult


@dataclass
class HandlerResult:
    """Result from a routing handler.

    Args:
        handled: Whether the handler resolved the request.
        result: Classification result.
        metadata: Optional metadata about handler decisions.
    """

    handled: bool
    result: ClassificationResult
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseHandler:
    """Base class for chain-of-responsibility handlers."""

    def __init__(self) -> None:
        self._next: Optional["BaseHandler"] = None

    def set_next(self, handler: "BaseHandler") -> "BaseHandler":
        """Set the next handler in the chain.

        Args:
            handler: Next handler.

        Returns:
            The handler passed in, for chaining.
        """

        self._next = handler
        return handler

    def handle(self, text: str, labels: Optional[List[str]] = None) -> HandlerResult:
        """Process a payload through the chain.

        Args:
            text: Issue text.
            labels: Optional candidate labels.

        Returns:
            HandlerResult containing the final decision.
        """

        handled = self._handle(text, labels)
        if handled.handled or self._next is None:
            return handled
        downstream = self._next.handle(text, labels)
        if handled.metadata:
            downstream.metadata = {**handled.metadata, **downstream.metadata}
        return downstream

    def _handle(self, text: str, labels: Optional[List[str]] = None) -> HandlerResult:
        """Handle a request at this node.

        Args:
            text: Issue text.
            labels: Optional candidate labels.

        Returns:
            HandlerResult for this node.
        """

        raise NotImplementedError


class LocalModelHandler(BaseHandler):
    """Handler that uses the local ML model for classification."""

    def __init__(self, classifier: Any, threshold: float) -> None:
        super().__init__()
        self.classifier = classifier
        self.threshold = threshold

    def _handle(self, text: str, labels: Optional[List[str]] = None) -> HandlerResult:
        """Handle using the local model and threshold.

        Args:
            text: Issue text.
            labels: Optional labels.

        Returns:
            HandlerResult with metadata about the local prediction.
        """

        result = self.classifier.predict(text, labels)
        metadata = {"local": {"label": result.label, "confidence": result.confidence}}
        if result.confidence >= self.threshold:
            return HandlerResult(handled=True, result=result, metadata=metadata)
        return HandlerResult(handled=False, result=result, metadata=metadata)


class LLMHandler(BaseHandler):
    """Handler that delegates classification to an LLM."""

    def __init__(self, classifier: Any) -> None:
        super().__init__()
        self.classifier = classifier

    def _handle(self, text: str, labels: Optional[List[str]] = None) -> HandlerResult:
        """Handle using the LLM classifier.

        Args:
            text: Issue text.
            labels: Optional labels.

        Returns:
            HandlerResult containing the LLM classification.
        """

        result = self.classifier.predict(text, labels)
        metadata = {"llm": {"label": result.label, "confidence": result.confidence}}
        return HandlerResult(handled=True, result=result, metadata=metadata)
