"""OpenRouter adapter for LLM fallback classification.

This module isolates OpenRouter HTTP calls and response parsing.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from requests import RequestException, Timeout

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class LLMResult:
    """Result container for LLM classification.

    Args:
        label: Predicted label.
        confidence: Confidence score in [0, 1].
        comment: Optional comment for triage response.
        raw: Raw model output or error details.
    """

    label: str
    confidence: float
    comment: str
    raw: str


def _truncate(text: str, limit: int = 400) -> str:
    """Truncate a string for logging.

    Args:
        text: Input string.
        limit: Maximum length.

    Returns:
        Truncated string.
    """

    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _format_response(response: requests.Response) -> str:
    """Format an HTTP response for error messages.

    Args:
        response: Requests response object.

    Returns:
        Summary string with status and body.
    """

    body = _truncate(response.text)
    return f"status={response.status_code} reason={response.reason} body={body}"


def _safe_json(response: requests.Response) -> Dict[str, Any]:
    """Parse JSON response safely.

    Args:
        response: Requests response object.

    Returns:
        Parsed JSON as dict.

    Raises:
        RuntimeError: If JSON parsing fails or payload is not a dict.
    """

    try:
        payload = response.json()
    except ValueError as exc:
        detail = _format_response(response)
        raise RuntimeError(f"Invalid JSON response: {detail}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("OpenRouter response JSON is not a dict")
    return payload


def _extract_content(payload: Dict[str, Any]) -> str:
    """Extract message content from the OpenRouter response.

    Args:
        payload: Parsed JSON payload.

    Returns:
        The model response content.

    Raises:
        ValueError: If required fields are missing.
    """

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter payload missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("OpenRouter choices entry is not a dict")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter message is not a dict")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("OpenRouter message content is missing")
    return content


class OpenRouterAdapter:
    """Adapter for OpenRouter chat completions API.

    Args:
        api_key: OpenRouter API key.
        model: Model identifier.
        base_url: Base API URL.
        timeout_seconds: Request timeout in seconds.
        fail_fast: Whether to raise on errors.

    Raises:
        ValueError: If the API key is missing.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api",
        timeout_seconds: int = 45,
        fail_fast: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter access")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.fail_fast = fail_fast
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost",
                "X-Title": "triage-agent",
            }
        )

    def classify_issue(self, text: str, labels: List[str]) -> LLMResult:
        """Classify an issue using the OpenRouter LLM.

        Args:
            text: Issue text.
            labels: Candidate labels.

        Returns:
            LLMResult with label, confidence, and comment.

        Raises:
            RuntimeError: If fail_fast is enabled and the request fails.
        """

        label_list = ", ".join(labels) if labels else "(no labels provided)"
        prompt = (
            "You are an operations triage assistant.\n"
            "Choose the single best label from the provided list.\n"
            "Return JSON with keys: label, confidence (0-1), comment.\n"
            "If none fit, choose the closest label.\n\n"
            f"Labels: {label_list}\n\n"
            f"Issue:\n{text}\n"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You classify GitHub issues."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        response: Optional[requests.Response] = None
        endpoint = f"{self.base_url}/v1/chat/completions"
        try:
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = _safe_json(response)
            content = _extract_content(data)
            parsed = self._parse_response(content)
            return LLMResult(
                label=str(parsed.get("label", "")),
                confidence=float(parsed.get("confidence", 0.0)),
                comment=str(parsed.get("comment", "")),
                raw=content,
            )
        except RequestException as exc:
            details = ""
            if getattr(exc, "response", None) is not None:
                details = _format_response(exc.response)
            if isinstance(exc, Timeout):
                message = (
                    f"OpenRouter request timed out after {self.timeout_seconds}s. "
                    f"model={self.model} endpoint={endpoint}. {details}"
                ).strip()
            else:
                message = (
                    f"OpenRouter request failed. model={self.model} "
                    f"endpoint={endpoint}. {details}"
                    if details
                    else f"OpenRouter request failed: {exc}"
                )
            if self.fail_fast:
                raise RuntimeError(message) from exc
            return LLMResult(label="", confidence=0.0, comment=message, raw=details)
        except (ValueError, RuntimeError) as exc:
            raw = _format_response(response) if response is not None else ""
            message = (
                f"OpenRouter response parsing failed: {exc}. {raw}"
                if raw
                else f"OpenRouter response parsing failed: {exc}"
            )
            if self.fail_fast:
                raise RuntimeError(message) from exc
            return LLMResult(label="", confidence=0.0, comment=message, raw=raw)

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from an LLM response string.

        Args:
            content: Raw model output.

        Returns:
            Parsed dict or empty dict if parsing fails.
        """

        content = content.strip()
        if content.startswith("{"):
            try:
                parsed = json.loads(content)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        match = JSON_RE.search(content)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}
