"""Dataset builder for training the local classifier."""

from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from triage.adapters.github import GitHubAdapter
from triage.preprocessing.text import sanitize_text


def _select_label(labels: List[str], allowlist: Optional[List[str]]) -> Optional[str]:
    """Select a label from a list, optionally using an allowlist.

    Args:
        labels: Candidate labels.
        allowlist: Optional allowlist of labels.

    Returns:
        Selected label or None if none match.
    """

    if allowlist:
        labels = [label for label in labels if label in allowlist]
    if not labels:
        return None
    return labels[0]


def build_dataset(
    adapter: GitHubAdapter,
    repos: Iterable[Tuple[str, str]],
    label_allowlist: Optional[List[str]] = None,
    max_issues_per_repo: Optional[int] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    fail_fast: bool = False,
) -> Tuple[List[str], List[str]]:
    """Build a training dataset from GitHub issues.

    Args:
        adapter: GitHub adapter.
        repos: Iterable of (owner, repo) tuples.
        label_allowlist: Optional list of labels to include.
        max_issues_per_repo: Optional max issues per repo.
        progress_cb: Optional callback for progress logging.
        fail_fast: Whether to raise on data issues.

    Returns:
        Tuple of (texts, labels).

    Raises:
        RuntimeError: If fail_fast is True and invalid data is found.
    """

    texts: List[str] = []
    targets: List[str] = []

    for owner, repo in repos:
        if progress_cb:
            progress_cb(f"Fetching issues from {owner}/{repo} ...")
        for issue in adapter.iter_issues(
            owner=owner,
            repo=repo,
            state="closed",
            max_issues=max_issues_per_repo,
        ):
            try:
                if not isinstance(issue, dict):
                    raise ValueError("Issue payload is not a dict")
                labels_payload = issue.get("labels", [])
                labels: List[str] = []
                if isinstance(labels_payload, list):
                    for label in labels_payload:
                        if isinstance(label, dict) and isinstance(label.get("name"), str):
                            labels.append(label["name"])
                label = _select_label(labels, label_allowlist)
                if not label:
                    if progress_cb:
                        progress_cb(
                            f"Skipping issue #{issue.get('number')} (no matching label)"
                        )
                    continue
                title = issue.get("title") if isinstance(issue.get("title"), str) else ""
                body = issue.get("body") if isinstance(issue.get("body"), str) else ""
                text = sanitize_text(f"{title}\n\n{body}")
                if not text:
                    if progress_cb:
                        progress_cb(
                            f"Skipping issue #{issue.get('number')} (empty after sanitize)"
                        )
                    continue
                texts.append(text)
                targets.append(label)
                if progress_cb and len(texts) % 25 == 0:
                    progress_cb(
                        f"Collected {len(texts)} samples so far "
                        f"(latest label: {label})"
                    )
            except (KeyError, TypeError, ValueError) as exc:
                if progress_cb:
                    progress_cb(
                        f"Skipping issue #{issue.get('number')} due to error: {exc}"
                    )
                if fail_fast:
                    raise RuntimeError("Invalid issue payload") from exc
                continue

    return texts, targets


def summarize_labels(labels: List[str]) -> Dict[str, int]:
    """Summarize label frequencies.

    Args:
        labels: List of labels.

    Returns:
        Mapping of label to count.
    """

    counts = Counter(labels)
    return dict(counts)
