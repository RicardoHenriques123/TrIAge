"""LLM classifier strategy wrapper."""

from typing import List, Optional

from triage.adapters.openrouter import OpenRouterAdapter
from triage.models.base import ClassificationResult


def _clamp(value: float) -> float:
    """Clamp a float to [0, 1]."""

    return max(0.0, min(1.0, value))


class LLMClassifier:
    """Classifier strategy that delegates to an LLM adapter.

    Args:
        adapter: OpenRouter adapter.
        fallback_label: Label to use if the model returns an invalid label.
    """

    def __init__(self, adapter: OpenRouterAdapter, fallback_label: str) -> None:
        self.adapter = adapter
        self.fallback_label = fallback_label

    def predict(self, text: str, labels: Optional[List[str]] = None) -> ClassificationResult:
        """Predict a label using an LLM.

        Args:
            text: Issue text.
            labels: Candidate labels.

        Returns:
            ClassificationResult with label and confidence.
        """

        labels = labels or []
        result = self.adapter.classify_issue(text, labels)
        label = result.label or self.fallback_label
        if labels and label not in labels:
            label = self.fallback_label
        confidence = _clamp(result.confidence)
        comment = result.comment
        source = "llm"
        lowered = comment.lower()
        if lowered.startswith("llm request failed") or lowered.startswith(
            "openrouter request failed"
        ) or lowered.startswith("openrouter response parsing failed"):
            source = "llm_error"
        return ClassificationResult(
            label=label,
            confidence=confidence,
            comment=comment,
            source=source,
        )
