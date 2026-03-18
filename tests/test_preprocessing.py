"""Tests for text preprocessing utilities."""

import pytest

from triage.preprocessing.text import sanitize_text


class TestSanitizeText:
    """Tests for sanitize_text function."""

    def test_empty_string(self):
        """Test that empty strings return empty."""
        assert sanitize_text("") == ""
        assert sanitize_text("   ") == ""

    def test_removes_code_blocks(self):
        """Test that code blocks are removed."""
        text = "Here's some code:\n```python\nprint('hello')\n```\nEnd."
        result = sanitize_text(text)
        assert "```" not in result
        assert "print" not in result
        assert "Here's some code:" in result
        assert "End." in result

    def test_removes_inline_code(self):
        """Test that inline code snippets are removed."""
        text = "Use the `foo()` function to do something."
        result = sanitize_text(text)
        assert "`" not in result
        assert "foo()" not in result
        assert "Use the" in result and "function" in result

    def test_extracts_markdown_links(self):
        """Test that markdown links are converted to text."""
        text = "Check out [this guide](https://example.com) for more."
        result = sanitize_text(text)
        assert "[" not in result
        assert "]" not in result
        assert "(" not in result
        assert "this guide" in result  # Link text extracted
        assert "Check out" in result
        assert "for more" in result

    def test_removes_images(self):
        """Test that image markdown is removed."""
        text = "See the screenshot: ![alt text](image.png) below."
        result = sanitize_text(text)
        assert "![" not in result
        assert "image.png" not in result
        assert "See the screenshot:" in result
        assert "below." in result

    def test_removes_html_tags(self):
        """Test that HTML tags are removed."""
        text = "This is <b>bold</b> and <i>italic</i> text."
        result = sanitize_text(text)
        assert "<" not in result
        assert ">" not in result
        assert "bold" in result
        assert "italic" in result

    def test_removes_blockquotes(self):
        """Test that blockquote markers are removed."""
        text = "> This is quoted.\n> More quotes."
        result = sanitize_text(text)
        assert ">" not in result
        assert "This is quoted." in result
        assert "More quotes." in result

    def test_removes_headings(self):
        """Test that heading markers are removed."""
        text = "# Main Title\n## Subtitle\nContent here."
        result = sanitize_text(text)
        assert "#" not in result
        assert "Main Title" in result
        assert "Subtitle" in result
        assert "Content here." in result

    def test_removes_lists(self):
        """Test that list markers are removed."""
        text = "- Item one\n- Item two\n1. Numbered\n2. List"
        result = sanitize_text(text)
        assert "-" not in result
        assert "1." not in result
        assert "Item one" in result
        assert "Item two" in result
        assert "Numbered" in result
        assert "List" in result

    def test_normalizes_whitespace(self):
        """Test that multiple whitespaces are normalized."""
        text = "Multiple   spaces    and\n\n\nline breaks."
        result = sanitize_text(text)
        assert "   " not in result
        assert "\n" not in result
        assert "Multiple spaces and line breaks." in result

    def test_handles_carriage_returns(self):
        """Test that carriage returns are handled."""
        text = "Line one\rLine two\r\nLine three"
        result = sanitize_text(text)
        assert "\r" not in result
        assert "Line one" in result
        assert "Line two" in result

    def test_complex_issue_text(self, sample_issue_text):
        """Test sanitization of real issue text."""
        result = sanitize_text(sample_issue_text)
        
        # Should preserve the core content
        assert "Bug" in result or "bug" in result.lower()
        assert "crashes" in result.lower()
        assert "startup" in result.lower()
        assert "Ubuntu" in result
        assert "Python" in result
        
        # Should remove markdown
        assert "#" not in result
        assert "**" not in result
        assert "-" not in result  # List markers removed
        
        # Result should be a single line or well-formatted
        assert "\n" not in result

    def test_preserves_actual_content(self):
        """Test that actual content is preserved through sanitization."""
        text = "Fix the bug where **users** cannot [log in](docs/login) to the app."
        result = sanitize_text(text)
        
        assert "Fix" in result
        assert "bug" in result
        assert "users" in result  # Bold markers removed or preserved, text kept
        assert "log in" in result  # Link text extracted
        assert "app" in result
        
        # Markdown artifacts for links should be gone
        assert "[" not in result
        assert "]" not in result or "log in" in result

    def test_edge_case_only_markdown(self):
        """Test text that is mostly markdown artifacts."""
        text = "# # # [link]()"
        result = sanitize_text(text)
        # Should be quite short after removing markdown
        assert len(result) < 20  # Much shorter than original

    def test_mixed_content(self):
        """Test mixed markdown and code content."""
        text = """# Issue: Performance

There's a performance issue in the `critical_function()`.

```python
def critical_function():
    pass
```

> As reported by user [John](https://github.com/john)

- Run profiler
- Fix bottleneck
"""
        result = sanitize_text(text)
        
        assert "#" not in result
        assert "```" not in result
        assert "`" not in result
        assert ">" not in result
        
        # Content should be preserved
        assert "Issue" in result or "Performance" in result
        assert "performance" in result.lower()
        assert "John" in result
        assert "profiler" in result
