"""Create and seed a demo GitHub repository for triage testing."""

import argparse
import logging
import sys
from typing import Dict, List, Optional

from triage.adapters.github import GitHubAdapter
from triage.config import load_config

logger = logging.getLogger(__name__)

LABELS: List[Dict[str, str]] = [
    {
        "name": "bug",
        "color": "d73a4a",
        "description": "Unexpected behavior or defect",
    },
    {
        "name": "feature",
        "color": "a2eeef",
        "description": "New functionality request",
    },
    {
        "name": "docs",
        "color": "0075ca",
        "description": "Documentation updates",
    },
    {
        "name": "support",
        "color": "fbca04",
        "description": "Support or usage questions",
    },
    {
        "name": "infra",
        "color": "5319e7",
        "description": "Infrastructure or CI",
    },
    {
        "name": "needs-triage",
        "color": "cfd3d7",
        "description": "Fallback triage label",
    },
]

SAMPLE_ISSUES: List[Dict[str, Optional[str]]] = [
    {
        "title": "Crash when saving profile with emoji",
        "body": "Steps to reproduce: update profile name with emoji and click save. App returns 500.",
        "label": "bug",
        "state": "closed",
    },
    {
        "title": "Webhook payload missing user id",
        "body": "The webhook payload does not include the user id field documented in v2.",
        "label": "bug",
        "state": "closed",
    },
    {
        "title": "Export button does nothing on Safari",
        "body": "In Safari 17, clicking export does not trigger a download.",
        "label": "bug",
        "state": "closed",
    },
    {
        "title": "Add bulk issue close action",
        "body": "We need a UI action to close multiple issues at once.",
        "label": "feature",
        "state": "closed",
    },
    {
        "title": "Support recurring schedule for agent",
        "body": "Please add a cron-like schedule for the triage agent.",
        "label": "feature",
        "state": "closed",
    },
    {
        "title": "Expose confidence score in API",
        "body": "We want confidence score included in the response metadata.",
        "label": "feature",
        "state": "closed",
    },
    {
        "title": "Docs: clarify setup steps",
        "body": "The README skips the environment variable setup for tokens.",
        "label": "docs",
        "state": "closed",
    },
    {
        "title": "Docs: add rate limit guidance",
        "body": "Include a note about GitHub rate limits and retry logic.",
        "label": "docs",
        "state": "closed",
    },
    {
        "title": "Docs: update API examples",
        "body": "Examples still reference deprecated endpoints.",
        "label": "docs",
        "state": "closed",
    },
    {
        "title": "How do I run this on a schedule?",
        "body": "Looking for the recommended cron schedule and flags.",
        "label": "support",
        "state": "closed",
    },
    {
        "title": "Can I use this with private repos?",
        "body": "Does the GitHub token need extra scopes for private repos?",
        "label": "support",
        "state": "closed",
    },
    {
        "title": "Need help with error 403",
        "body": "Agent exits with 403 even though token works elsewhere.",
        "label": "support",
        "state": "closed",
    },
    {
        "title": "CI fails on python 3.12",
        "body": "Tests fail with numpy build errors on 3.12 runners.",
        "label": "infra",
        "state": "closed",
    },
    {
        "title": "Add lint step to pipeline",
        "body": "We should add ruff or flake8 to CI to enforce style.",
        "label": "infra",
        "state": "closed",
    },
    {
        "title": "Deploy job timing out",
        "body": "Pipeline deploy job times out after 10 minutes in staging.",
        "label": "infra",
        "state": "closed",
    },
    {
        "title": "Agent sometimes skips issues",
        "body": "Noticed some open issues remain unlabeled after runs.",
        "label": None,
        "state": "open",
    },
    {
        "title": "Triage labels are inconsistent",
        "body": "We see mismatched labels between repos. Need help.",
        "label": None,
        "state": "open",
    },
    {
        "title": "Webhook integration question",
        "body": "Can this run as a webhook instead of a cron?",
        "label": None,
        "state": "open",
    },
    {
        "title": "Add a demo repo",
        "body": "We need a demo repo with sample issues for onboarding.",
        "label": None,
        "state": "open",
    },
    {
        "title": "Label suggestion",
        "body": "Should we add a label for billing issues?",
        "label": None,
        "state": "open",
    },
]


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configure a console logger.

    Args:
        verbose: Whether to enable debug logging.

    Returns:
        Configured logger.
    """

    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def parse_repo(value: str) -> Dict[str, str]:
    """Parse a repo string into owner and name.

    Args:
        value: Repository string in owner/name format.

    Returns:
        Dict with owner and name.

    Raises:
        ValueError: If the repo string is invalid.
    """

    if "/" not in value:
        raise ValueError("--repo must be in owner/name format")
    owner, name = value.split("/", 1)
    return {"owner": owner.strip(), "name": name.strip()}


def ensure_labels(
    adapter: GitHubAdapter, owner: str, repo: str, fail_fast: bool = False
) -> None:
    """Ensure demo labels exist in the repository.

    Args:
        adapter: GitHub adapter.
        owner: Repository owner.
        repo: Repository name.
        fail_fast: Whether to raise on failures.

    Raises:
        RuntimeError: If fail_fast is enabled and label creation fails.
    """

    existing = set(adapter.list_labels(owner, repo))
    for label in LABELS:
        if label["name"] in existing:
            continue
        try:
            adapter.create_label(
                owner,
                repo,
                name=label["name"],
                color=label["color"],
                description=label["description"],
            )
        except RuntimeError as exc:
            logger.warning(
                "Failed to create label '%s' for %s/%s: %s",
                label["name"],
                owner,
                repo,
                exc,
            )
            if fail_fast:
                raise


def seed_issues(
    adapter: GitHubAdapter, owner: str, repo: str, fail_fast: bool = False
) -> None:
    """Seed issues into the demo repository.

    Args:
        adapter: GitHub adapter.
        owner: Repository owner.
        repo: Repository name.
        fail_fast: Whether to raise on failures.

    Raises:
        RuntimeError: If fail_fast is enabled and issue creation fails.
    """

    for issue in SAMPLE_ISSUES:
        labels = [issue["label"]] if issue.get("label") else None
        try:
            created = adapter.create_issue(
                owner=owner,
                repo=repo,
                title=issue["title"],
                body=issue["body"],
                labels=labels,
            )
            if issue["state"] == "closed":
                adapter.update_issue(
                    owner=owner,
                    repo=repo,
                    number=created["number"],
                    state="closed",
                )
        except RuntimeError as exc:
            logger.warning(
                "Failed to seed issue '%s' for %s/%s: %s",
                issue["title"],
                owner,
                repo,
                exc,
            )
            if fail_fast:
                raise


def main() -> None:
    """CLI entrypoint for demo repo creation."""

    parser = argparse.ArgumentParser(
        description="Create and seed a demo GitHub repository with sample issues for testing.",
        epilog="For detailed setup instructions, see README.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Repository name (owner/name format).",
    )
    parser.add_argument(
        "--org",
        default=None,
        help="Organization name (uses repo owner if omitted).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create as private repository.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Seed issues/labels to existing repo instead of failing.",
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Create repo without seeding labels and issues.",
    )
    parser.add_argument(
        "--description",
        default="Demo repo for triage automation",
        help="Repository description.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output and full error tracebacks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    
    try:
        args = parser.parse_args()
    except SystemExit as e:
        if e.code != 0:
            # argparse prints its own error message
            pass
        raise

    logger = setup_logger(verbose=args.verbose)

    try:
        config = load_config()
        if not config.github_token:
            raise RuntimeError("GITHUB_TOKEN is required.")

        try:
            repo_info = parse_repo(args.repo)
        except ValueError as e:
            logger.error("Invalid repository format: %s", e)
            logger.error("Usage: --repo owner/name")
            parser.print_usage()
            raise SystemExit(1) from e
        
        owner = args.org if args.org else repo_info["owner"]
        name = repo_info["name"]

        adapter = GitHubAdapter(token=config.github_token, api_url=config.github_api_url)
        existing_repo = adapter.get_repo(owner, name)

        if existing_repo is None:
            adapter.create_repo(
                name=name,
                private=args.private,
                org=args.org,
                description=args.description,
            )
        elif not args.allow_existing:
            raise RuntimeError(
                "Repo already exists. Use --allow-existing to seed labels/issues."
            )

        if not args.skip_seed:
            ensure_labels(adapter, owner, name, fail_fast=args.debug)
            seed_issues(adapter, owner, name, fail_fast=args.debug)

        logger.info("Demo repo ready: %s/%s", owner, name)
    except (RuntimeError, ValueError, OSError) as exc:
        if args.debug:
            raise
        logger.error("Demo repo creation failed: %s", exc)
        logger.error("Re-run with --debug to see the full traceback.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
