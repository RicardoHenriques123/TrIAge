"""GitHub API adapter for issue triage.

This module isolates GitHub REST API calls behind a small, typed interface
so core triage logic remains decoupled from HTTP details.
"""

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, Iterable, List, Optional

import requests
from requests import RequestException

logger = logging.getLogger(__name__)


@dataclass
class GitHubIssue:
    """Normalized GitHub issue payload used by the triage system.

    Args:
        owner: Repository owner.
        repo: Repository name.
        number: Issue number.
        title: Issue title.
        body: Issue body text.
        labels: List of label names.
        url: HTML URL for the issue.
    """

    owner: str
    repo: str
    number: int
    title: str
    body: str
    labels: List[str]
    url: str

    @property
    def text(self) -> str:
        """Return the concatenated issue title and body for classification."""

        return f"{self.title}\n\n{self.body or ''}".strip()


class GitHubAdapter:
    """Adapter that wraps GitHub REST API interactions.

    Args:
        token: GitHub Personal Access Token.
        api_url: Base GitHub API URL.

    Raises:
        ValueError: If the token is missing.
    """

    def __init__(self, token: str, api_url: str = "https://api.github.com") -> None:
        if not token:
            raise ValueError("GITHUB_TOKEN is required for GitHub API access")
        self.token = token
        self.api_url = api_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "triage-agent",
            }
        )

    def _format_response(self, response: requests.Response) -> str:
        """Format a response for error messages.

        Args:
            response: Requests response object.

        Returns:
            A short string with status, reason, and truncated body.
        """

        body = response.text.strip()
        if len(body) > 400:
            body = body[:400] + "..."
        return f"status={response.status_code} reason={response.reason} body={body}"

    def _safe_json(self, response: requests.Response) -> Any:
        """Parse JSON from a response safely.

        Args:
            response: Requests response object.

        Returns:
            Parsed JSON payload.

        Raises:
            RuntimeError: If JSON parsing fails.
        """

        try:
            return response.json()
        except ValueError as exc:
            detail = self._format_response(response)
            raise RuntimeError(f"Invalid JSON response: {detail}") from exc

    def _request(
        self, method: str, path: str, allow_not_found: bool = False, **kwargs: Any
    ) -> requests.Response:
        """Perform an HTTP request to GitHub with basic error handling.

        Args:
            method: HTTP method.
            path: API path, e.g., `/repos/{owner}/{repo}`.
            allow_not_found: Whether to allow HTTP 404 responses.
            **kwargs: Requests options.

        Returns:
            The HTTP response.

        Raises:
            RuntimeError: If the request fails or returns an error status.
        """

        url = f"{self.api_url}{path}"
        response: Optional[requests.Response] = None
        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            if response.status_code == 403 and "rate limit" in response.text.lower():
                reset = response.headers.get("X-RateLimit-Reset")
                if reset:
                    sleep_for = max(0, int(reset) - int(time.time()))
                    time.sleep(min(sleep_for, 5))
            if response.status_code == 404 and allow_not_found:
                return response
            response.raise_for_status()
            return response
        except RequestException as exc:
            details = ""
            if getattr(exc, "response", None) is not None:
                details = self._format_response(exc.response)
            elif response is not None:
                details = self._format_response(response)
            message = (
                f"GitHub API request failed: {method} {url}. {details}"
                if details
                else f"GitHub API request failed: {method} {url}"
            )
            raise RuntimeError(message) from exc

    def iter_issues(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        per_page: int = 100,
        max_issues: Optional[int] = None,
        labels: Optional[str] = None,
    ) -> Iterable[Dict[str, Any]]:
        """Iterate over issues in a repository.

        Args:
            owner: Repository owner.
            repo: Repository name.
            state: Issue state filter.
            per_page: Page size.
            max_issues: Optional maximum number of issues to yield.
            labels: Optional comma-separated label filter.

        Yields:
            Raw issue payloads from the GitHub API.

        Raises:
            RuntimeError: If the API returns invalid data.
        """

        page = 1
        seen = 0
        while True:
            params: Dict[str, Any] = {
                "state": state,
                "per_page": per_page,
                "page": page,
                "sort": "updated",
                "direction": "desc",
            }
            if labels:
                params["labels"] = labels
            response = self._request(
                "GET", f"/repos/{owner}/{repo}/issues", params=params
            )
            payload = self._safe_json(response)
            if not isinstance(payload, list):
                raise RuntimeError("GitHub issues payload is not a list")
            if not payload:
                break
            for issue in payload:
                if not isinstance(issue, dict):
                    logger.warning("Skipping non-dict issue payload")
                    continue
                if "pull_request" in issue:
                    continue
                yield issue
                seen += 1
                if max_issues and seen >= max_issues:
                    return
            page += 1

    def list_labels(self, owner: str, repo: str) -> List[str]:
        """List label names for a repository.

        Args:
            owner: Repository owner.
            repo: Repository name.

        Returns:
            List of label names.
        """

        labels: List[str] = []
        page = 1
        while True:
            response = self._request(
                "GET",
                f"/repos/{owner}/{repo}/labels",
                params={"per_page": 100, "page": page},
            )
            payload = self._safe_json(response)
            if not isinstance(payload, list):
                raise RuntimeError("GitHub labels payload is not a list")
            if not payload:
                break
            for label in payload:
                if isinstance(label, dict) and isinstance(label.get("name"), str):
                    labels.append(label["name"])
            page += 1
        return labels

    def add_labels(self, owner: str, repo: str, number: int, labels: List[str]) -> None:
        """Apply labels to an issue.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Issue number.
            labels: Label names to apply.
        """

        self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{number}/labels",
            json={"labels": labels},
        )

    def create_comment(self, owner: str, repo: str, number: int, body: str) -> None:
        """Create a comment on an issue.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Issue number.
            body: Comment body.
        """

        self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{number}/comments",
            json={"body": body},
        )

    def get_repo(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Fetch repository metadata.

        Args:
            owner: Repository owner.
            repo: Repository name.

        Returns:
            Repository payload or None if not found.
        """

        response = self._request("GET", f"/repos/{owner}/{repo}", allow_not_found=True)
        if response.status_code == 404:
            return None
        payload = self._safe_json(response)
        if not isinstance(payload, dict):
            raise RuntimeError("GitHub repo payload is not a dict")
        return payload

    def create_repo(
        self,
        name: str,
        private: bool = False,
        org: Optional[str] = None,
        description: str = "",
        auto_init: bool = True,
    ) -> Dict[str, Any]:
        """Create a new repository.

        Args:
            name: Repository name.
            private: Whether the repo should be private.
            org: Optional organization name.
            description: Repo description.
            auto_init: Whether to initialize with a README.

        Returns:
            Repository payload.
        """

        payload = {
            "name": name,
            "private": private,
            "description": description,
            "has_issues": True,
            "auto_init": auto_init,
        }
        if org:
            response = self._request("POST", f"/orgs/{org}/repos", json=payload)
        else:
            response = self._request("POST", "/user/repos", json=payload)
        result = self._safe_json(response)
        if not isinstance(result, dict):
            raise RuntimeError("GitHub repo creation payload is not a dict")
        return result

    def create_label(
        self,
        owner: str,
        repo: str,
        name: str,
        color: str = "ededed",
        description: str = "",
    ) -> Dict[str, Any]:
        """Create a label in a repository.

        Args:
            owner: Repository owner.
            repo: Repository name.
            name: Label name.
            color: Hex color without '#'.
            description: Label description.

        Returns:
            Label payload.
        """

        response = self._request(
            "POST",
            f"/repos/{owner}/{repo}/labels",
            json={"name": name, "color": color, "description": description},
        )
        payload = self._safe_json(response)
        if not isinstance(payload, dict):
            raise RuntimeError("GitHub label payload is not a dict")
        return payload

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str = "",
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new issue.

        Args:
            owner: Repository owner.
            repo: Repository name.
            title: Issue title.
            body: Issue body.
            labels: Optional list of labels.

        Returns:
            Issue payload.
        """

        payload: Dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        response = self._request(
            "POST", f"/repos/{owner}/{repo}/issues", json=payload
        )
        result = self._safe_json(response)
        if not isinstance(result, dict):
            raise RuntimeError("GitHub issue payload is not a dict")
        return result

    def update_issue(
        self,
        owner: str,
        repo: str,
        number: int,
        state: Optional[str] = None,
        title: Optional[str] = None,
        body: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an issue.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Issue number.
            state: New issue state.
            title: New title.
            body: New body.

        Returns:
            Updated issue payload.
        """

        payload: Dict[str, Any] = {}
        if state:
            payload["state"] = state
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        response = self._request(
            "PATCH", f"/repos/{owner}/{repo}/issues/{number}", json=payload
        )
        result = self._safe_json(response)
        if not isinstance(result, dict):
            raise RuntimeError("GitHub update payload is not a dict")
        return result

    def to_issue(self, owner: str, repo: str, issue_payload: Dict[str, Any]) -> GitHubIssue:
        """Convert a raw GitHub API payload into a GitHubIssue.

        Args:
            owner: Repository owner.
            repo: Repository name.
            issue_payload: Raw issue payload.

        Returns:
            Normalized GitHubIssue object.

        Raises:
            ValueError: If the payload is missing required fields.
        """

        if not isinstance(issue_payload, dict):
            raise ValueError("Issue payload must be a dict")

        number = issue_payload.get("number")
        if not isinstance(number, int):
            raise ValueError("Issue payload is missing a valid number")

        title = issue_payload.get("title")
        body = issue_payload.get("body")
        title = title if isinstance(title, str) else ""
        body = body if isinstance(body, str) else ""

        labels_payload = issue_payload.get("labels", [])
        labels: List[str] = []
        if isinstance(labels_payload, list):
            for label in labels_payload:
                if isinstance(label, dict) and isinstance(label.get("name"), str):
                    labels.append(label["name"])

        url = issue_payload.get("html_url")
        url = url if isinstance(url, str) else ""

        return GitHubIssue(
            owner=owner,
            repo=repo,
            number=number,
            title=title,
            body=body,
            labels=labels,
            url=url,
        )
