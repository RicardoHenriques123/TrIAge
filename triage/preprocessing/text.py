"""Text preprocessing utilities for issue triage."""

import re

CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]+`")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
BLOCKQUOTE_RE = re.compile(r"^>+\s*", re.MULTILINE)
HEADING_RE = re.compile(r"^#+\s+", re.MULTILINE)
LIST_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
NUMBERED_LIST_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
WHITESPACE_RE = re.compile(r"\s+")


def sanitize_text(text: str) -> str:
    """Remove Markdown artifacts and normalize whitespace.

    Args:
        text: Raw issue text.

    Returns:
        Sanitized text.
    """

    if not text:
        return ""

    cleaned = text
    cleaned = CODE_BLOCK_RE.sub(" ", cleaned)
    cleaned = IMAGE_RE.sub(" ", cleaned)
    cleaned = MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = INLINE_CODE_RE.sub(" ", cleaned)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    cleaned = BLOCKQUOTE_RE.sub("", cleaned)
    cleaned = HEADING_RE.sub("", cleaned)
    cleaned = LIST_RE.sub("", cleaned)
    cleaned = NUMBERED_LIST_RE.sub("", cleaned)
    cleaned = cleaned.replace("\r", " ")
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()
