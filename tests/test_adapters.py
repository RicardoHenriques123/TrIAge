"""Tests for API adapters."""

import pytest
from unittest.mock import MagicMock, patch
from requests.exceptions import RequestException

from triage.adapters.github import GitHubAdapter, GitHubIssue
from triage.adapters.openrouter import OpenRouterAdapter


class TestGitHubIssue:
    """Tests for GitHubIssue dataclass."""

    def test_creation(self):
        """Test creating a GitHubIssue."""
        issue = GitHubIssue(
            owner="owner",
            repo="repo",
            number=42,
            title="Bug title",
            body="Bug body",
            labels=["bug", "critical"],
            url="https://github.com/owner/repo/issues/42"
        )
        
        assert issue.owner == "owner"
        assert issue.repo == "repo"
        assert issue.number == 42
        assert issue.title == "Bug title"

    def test_text_property_combines_title_and_body(self):
        """Test that text property combines title and body."""
        issue = GitHubIssue(
            owner="owner",
            repo="repo",
            number=1,
            title="Title here",
            body="Body content",
            labels=[],
            url="https://example.com"
        )
        
        assert issue.text == "Title here\n\nBody content"

    def test_text_property_handles_empty_body(self):
        """Test text property when body is empty."""
        issue = GitHubIssue(
            owner="owner",
            repo="repo",
            number=1,
            title="Just title",
            body="",
            labels=[],
            url="https://example.com"
        )
        
        assert issue.text == "Just title"

    def test_text_property_handles_none_body(self):
        """Test text property when body is None or missing."""
        issue = GitHubIssue(
            owner="owner",
            repo="repo",
            number=1,
            title="Title",
            body=None,
            labels=[],
            url="https://example.com"
        )
        
        assert "Title" in issue.text


class TestGitHubAdapter:
    """Tests for GitHubAdapter."""

    def test_initialization_with_token(self):
        """Test initializing adapter with valid token."""
        adapter = GitHubAdapter(token="ghp_test123", api_url="https://api.github.com")
        
        assert adapter.token == "ghp_test123"
        assert adapter.api_url == "https://api.github.com"
        assert adapter.session is not None

    def test_initialization_without_token_raises(self):
        """Test that missing token raises ValueError."""
        with pytest.raises(ValueError):
            GitHubAdapter(token="", api_url="https://api.github.com")

    def test_initialization_strips_trailing_slash(self):
        """Test that API URL trailing slashes are removed."""
        adapter = GitHubAdapter(token="ghp_test", api_url="https://api.github.com/")
        assert adapter.api_url == "https://api.github.com"

    def test_session_headers_set(self):
        """Test that session headers are properly configured."""
        adapter = GitHubAdapter(token="ghp_mytoken")
        
        assert "Authorization" in adapter.session.headers
        assert adapter.session.headers["Authorization"] == "token ghp_mytoken"
        assert "Accept" in adapter.session.headers
        assert "User-Agent" in adapter.session.headers

    def test_to_issue_conversion(self):
        """Test converting GitHub API response to GitHubIssue."""
        adapter = GitHubAdapter(token="ghp_test")
        
        payload = {
            "number": 42,
            "title": "Test issue",
            "body": "Test body",
            "labels": [{"name": "bug"}, {"name": "high"}],
            "html_url": "https://github.com/test/repo/issues/42"
        }
        
        issue = adapter.to_issue("owner", "repo", payload)
        
        assert isinstance(issue, GitHubIssue)
        assert issue.number == 42
        assert issue.title == "Test issue"
        assert issue.labels == ["bug", "high"]

    def test_format_response_truncates_long_body(self):
        """Test that response formatting truncates long bodies."""
        adapter = GitHubAdapter(token="ghp_test")
        
        response = MagicMock()
        response.status_code = 400
        response.reason = "Bad Request"
        response.text = "x" * 500  # Long text
        
        formatted = adapter._format_response(response)
        
        assert "400" in formatted
        assert "Bad Request" in formatted
        assert len(formatted) < len(response.text)  # Should be truncated
        assert "..." in formatted  # Truncation marker


class TestOpenRouterAdapter:
    """Tests for OpenRouterAdapter."""

    def test_initialization_with_key(self):
        """Test initializing adapter with valid API key."""
        adapter = OpenRouterAdapter(
            api_key="sk_test123",
            model="openai/gpt-4o-mini"
        )
        
        assert adapter.api_key == "sk_test123"
        assert adapter.model == "openai/gpt-4o-mini"
        assert adapter.timeout_seconds == 45  # Default

    def test_initialization_without_key_raises(self):
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError):
            OpenRouterAdapter(api_key="", model="openai/gpt-4o-mini")

    def test_initialization_custom_timeout(self):
        """Test setting custom timeout."""
        adapter = OpenRouterAdapter(
            api_key="sk_test",
            model="openai/gpt-4o-mini",
            timeout_seconds=120
        )
        
        assert adapter.timeout_seconds == 120

    def test_session_headers_set(self):
        """Test that session headers are properly configured."""
        adapter = OpenRouterAdapter(api_key="sk_mykey", model="openai/gpt-4o-mini")
        
        assert "Authorization" in adapter.session.headers
        assert adapter.session.headers["Authorization"] == "Bearer sk_mykey"
        assert "Content-Type" in adapter.session.headers
        assert adapter.session.headers["Content-Type"] == "application/json"

    @patch("triage.adapters.openrouter.requests.Session.post")
    def test_classify_issue_api_call(self, mock_post):
        """Test that classify_issue makes correct API calls."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"label": "bug", "confidence": 0.92, "comment": "Found bug"}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        adapter = OpenRouterAdapter(api_key="sk_test", model="openai/gpt-4o-mini")
        adapter.session.post = mock_post
        
        result = adapter.classify_issue("Issue text", labels=["bug", "feature"])
        
        # Result is LLMResult, not ClassificationResult at adapter level
        assert result is not None
        assert result.label == "bug"
        mock_post.assert_called_once()

    def test_classify_issue_error_handling(self):
        """Test that classify_issue handles API errors gracefully."""
        adapter = OpenRouterAdapter(api_key="sk_test", model="openai/gpt-4o-mini", fail_fast=False)
        
        with patch.object(adapter.session, "post") as mock_post:
            mock_post.side_effect = RequestException("Connection failed")
            
            result = adapter.classify_issue("Issue text", labels=[])
            
            # LLMResult has comment field for errors
            assert "failed" in result.comment.lower() or result.label == ""

    def test_classify_issue_timeout_handling(self):
        """Test that timeouts are handled properly."""
        adapter = OpenRouterAdapter(
            api_key="sk_test",
            model="openai/gpt-4o-mini",
            timeout_seconds=1,
            fail_fast=False
        )
        
        with patch.object(adapter.session, "post") as mock_post:
            from requests.exceptions import Timeout
            mock_post.side_effect = Timeout("Request timeout")
            
            result = adapter.classify_issue("Issue text", labels=[])
            
            assert "timeout" in result.comment.lower() or result.label == ""

    def test_classify_issue_with_empty_labels(self):
        """Test classify_issue with no candidate labels."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"label": "bug", "confidence": 0.85, "comment": "Bug found"}'
                    }
                }
            ]
        }
        
        adapter = OpenRouterAdapter(api_key="sk_test", model="openai/gpt-4o-mini")
        
        with patch.object(adapter.session, "post") as mock_post:
            mock_post.return_value = mock_response
            result = adapter.classify_issue("Issue text", labels=None)
            
            assert result.label == "bug"

    def test_classify_issue_invalid_json_response(self):
        """Test handling of invalid JSON response."""
        adapter = OpenRouterAdapter(
            api_key="sk_test",
            model="openai/gpt-4o-mini",
            fail_fast=False
        )
        
        with patch.object(adapter.session, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_post.return_value = mock_response
            
            result = adapter.classify_issue("Issue text", labels=[])
            
            assert "invalid" in result.comment.lower() or "json" in result.comment.lower() or result.label == ""

    def test_classify_issue_malformed_response_structure(self):
        """Test handling of malformed response structure."""
        adapter = OpenRouterAdapter(
            api_key="sk_test",
            model="openai/gpt-4o-mini",
            fail_fast=False
        )
        
        with patch.object(adapter.session, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            # Missing required fields
            mock_response.json.return_value = {"invalid": "structure"}
            mock_post.return_value = mock_response
            
            result = adapter.classify_issue("Issue text", labels=[])
            
            # Should have error in comment when structure is wrong
            assert result.label == "" or "error" in result.comment.lower()
