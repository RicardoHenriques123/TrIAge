"""Base model interfaces for classification strategies."""

from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class ClassificationResult:
    """Result of a classification call.

    Args:
        label: Predicted label.
        confidence: Confidence score.
        comment: Optional comment for triage response.
        source: Source identifier (local, llm, etc.).
    """

    label: str
    confidence: float
    comment: str = ""
    source: str = ""


class ClassifierStrategy(Protocol):
    """Protocol for classifier strategies used by the router."""

    def predict(self, text: str, labels: Optional[List[str]] = None) -> ClassificationResult:
        """Predict a label for the given text.

        Args:
            text: Input text.
            labels: Optional candidate labels.

        Returns:
            ClassificationResult with predicted label.
        """

        ...
