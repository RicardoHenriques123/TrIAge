"""Local ML model wrapper for issue classification."""

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np

from triage.models.base import ClassificationResult


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities for a score vector.

    Args:
        scores: Score vector.

    Returns:
        Probability vector.
    """

    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


class LocalModelClassifier:
    """Wrapper for the serialized scikit-learn pipeline.

    Args:
        model_path: Path to the serialized model file.

    Raises:
        FileNotFoundError: If the model file is missing.
        RuntimeError: If the model cannot be loaded.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        try:
            self.pipeline = joblib.load(self.model_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError("Failed to load local model") from exc

    def predict(self, text: str, labels: Optional[List[str]] = None) -> ClassificationResult:
        """Predict a label for an issue text.

        Args:
            text: Issue text.
            labels: Optional candidate labels (unused for local model).

        Returns:
            ClassificationResult with label and confidence.
        """

        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba([text])[0]
            label_index = int(np.argmax(probabilities))
            confidence = float(probabilities[label_index])
            classes = list(getattr(self.pipeline, "classes_", []))
            if classes:
                label = str(classes[label_index])
            else:
                label = str(self.pipeline.predict([text])[0])
            return ClassificationResult(label=label, confidence=confidence, source="local")

        if hasattr(self.pipeline, "decision_function"):
            scores = np.array(self.pipeline.decision_function([text]))
            if scores.ndim == 1:
                scalar = float(scores[0]) if scores.size else 0.0
                probabilities = _softmax(np.array([-scalar, scalar]))
            else:
                probabilities = _softmax(np.array(scores[0]))
            confidence = float(np.max(probabilities))
            label = str(self.pipeline.predict([text])[0])
            return ClassificationResult(label=label, confidence=confidence, source="local")

        prediction = self.pipeline.predict([text])[0]
        return ClassificationResult(label=str(prediction), confidence=0.0, source="local")
