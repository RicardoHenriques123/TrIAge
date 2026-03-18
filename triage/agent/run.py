"""Online inference agent for GitHub issue triage."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from triage.adapters.github import GitHubAdapter
from triage.adapters.openrouter import OpenRouterAdapter
from triage.config import load_config
from triage.models.llm import LLMClassifier
from triage.models.local_model import LocalModelClassifier
from triage.preprocessing.text import sanitize_text
from triage.routing.handlers import BaseHandler, LLMHandler, LocalModelHandler
from triage.telemetry.logger import TelemetryLogger


def setup_logger() -> logging.Logger:
    """Configure a console logger for the agent.

    Returns:
        Configured logger.
    """

    logger = logging.getLogger("triage.agent")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def parse_repos(values: List[str]) -> List[Tuple[str, str]]:
    """Parse owner/repo strings into tuples.

    Args:
        values: List of repo strings.

    Returns:
        List of (owner, repo) tuples.

    Raises:
        ValueError: If any repo is malformed.
    """

    repos: List[Tuple[str, str]] = []
    for value in values:
        if "/" not in value:
            raise ValueError(f"Invalid repo '{value}', expected owner/repo")
        owner, repo = value.split("/", 1)
        repos.append((owner.strip(), repo.strip()))
    return repos


def build_chain(
    model_path: Path,
    threshold: float,
    llm_classifier: Optional[LLMClassifier],
) -> BaseHandler:
    """Build the routing chain for hybrid classification.

    Args:
        model_path: Local model path.
        threshold: Confidence threshold.
        llm_classifier: Optional LLM classifier.

    Returns:
        BaseHandler for the chain.
    """

    local_handler = LocalModelHandler(LocalModelClassifier(model_path), threshold)
    if llm_classifier:
        local_handler.set_next(LLMHandler(llm_classifier))
    return local_handler


def main() -> None:
    """CLI entrypoint for the inference agent."""

    parser = argparse.ArgumentParser(description="Run triage agent on GitHub issues")
    parser.add_argument("--repos", nargs="+", required=True, help="owner/repo list")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-issues", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--comment-on-high-confidence", action="store_true")
    parser.add_argument("--label-allowlist", nargs="*", default=None)
    parser.add_argument("--allow-no-llm", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    console_logger = setup_logger()

    try:
        config = load_config()
        dry_run = args.dry_run or config.dry_run

        if not config.github_token:
            raise RuntimeError("GITHUB_TOKEN is required.")

        adapter = GitHubAdapter(token=config.github_token, api_url=config.github_api_url)
        repos = parse_repos(args.repos)

        if not config.openrouter_api_key and not args.allow_no_llm:
            raise RuntimeError("OPENROUTER_API_KEY is required for LLM fallback.")

        llm_classifier: Optional[LLMClassifier] = None
        if config.openrouter_api_key and not args.allow_no_llm:
            llm_adapter = OpenRouterAdapter(
                api_key=config.openrouter_api_key,
                model=config.openrouter_model,
                timeout_seconds=config.openrouter_timeout,
                fail_fast=args.debug,
            )
            llm_classifier = LLMClassifier(
                llm_adapter, fallback_label=config.fallback_label
            )

        model_path = Path(args.model_path) if args.model_path else config.model_path
        threshold = (
            args.threshold if args.threshold is not None else config.confidence_threshold
        )

        handler_chain = build_chain(
            model_path=model_path, threshold=threshold, llm_classifier=llm_classifier
        )
        telemetry = TelemetryLogger(config.log_path)

        for owner, repo in repos:
            repo_counts: Dict[str, int] = {
                "fetched": 0,
                "processed": 0,
                "skipped_labeled": 0,
                "skipped_low_confidence": 0,
                "errors": 0,
            }
            try:
                label_candidates = args.label_allowlist or adapter.list_labels(
                    owner, repo
                )
            except RuntimeError as exc:
                console_logger.error(
                    "Failed to fetch labels for %s/%s: %s", owner, repo, exc
                )
                telemetry.log_event(
                    "error",
                    {
                        "owner": owner,
                        "repo": repo,
                        "error": str(exc),
                        "stage": "list_labels",
                    },
                )
                if args.debug:
                    raise
                continue

            for issue_payload in adapter.iter_issues(
                owner=owner,
                repo=repo,
                state="open",
                max_issues=args.max_issues,
            ):
                try:
                    repo_counts["fetched"] += 1
                    issue = adapter.to_issue(owner, repo, issue_payload)
                    if issue.labels:
                        repo_counts["skipped_labeled"] += 1
                        continue
                    start_time = time.time()
                    cleaned_text = sanitize_text(issue.text)
                    route_result = handler_chain.handle(
                        cleaned_text, labels=label_candidates
                    )
                    elapsed = time.time() - start_time
                    local_meta = route_result.metadata.get("local", {})
                    local_label = local_meta.get("label", "")
                    local_confidence = local_meta.get("confidence", None)
                    if not route_result.handled:
                        repo_counts["skipped_low_confidence"] += 1
                        telemetry.log_decision(
                            {
                                "owner": owner,
                                "repo": repo,
                                "issue_number": issue.number,
                                "issue_url": issue.url,
                                "label": "",
                                "confidence": route_result.result.confidence,
                                "source": route_result.result.source,
                                "local_label": local_label,
                                "local_confidence": local_confidence,
                                "duration_seconds": round(elapsed, 4),
                                "dry_run": dry_run,
                                "skipped": True,
                                "reason": "low_confidence_no_llm",
                            }
                        )
                        continue

                    label = route_result.result.label or config.fallback_label
                    if label_candidates:
                        if label not in label_candidates:
                            if config.fallback_label in label_candidates:
                                label = config.fallback_label
                            else:
                                label = label_candidates[0]
                    comment = route_result.result.comment
                    llm_error = route_result.result.source == "llm_error"
                    error_message = comment if llm_error else ""
                    if llm_error and comment:
                        console_logger.warning(
                            "LLM fallback failed for %s/%s#%s: %s",
                            owner,
                            repo,
                            issue.number,
                            comment,
                        )
                        telemetry.log_event(
                            "llm_error",
                            {
                                "owner": owner,
                                "repo": repo,
                                "issue_number": issue.number,
                                "error": comment,
                            },
                        )
                        comment = ""

                    if not dry_run:
                        adapter.add_labels(owner, repo, issue.number, [label])
                        if (comment or args.comment_on_high_confidence) and not llm_error:
                            body = comment or (
                                f"Auto-triage applied label `{label}` with confidence "
                                f"{route_result.result.confidence:.2f}."
                            )
                            adapter.create_comment(owner, repo, issue.number, body)

                    repo_counts["processed"] += 1
                    telemetry.log_decision(
                        {
                            "owner": owner,
                            "repo": repo,
                            "issue_number": issue.number,
                            "issue_url": issue.url,
                            "label": label,
                            "confidence": route_result.result.confidence,
                            "source": route_result.result.source,
                            "local_label": local_label,
                            "local_confidence": local_confidence,
                            "duration_seconds": round(elapsed, 4),
                            "dry_run": dry_run,
                            "skipped": False,
                            "error": error_message,
                        }
                    )
                except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                    console_logger.error(
                        "Failed to process issue in %s/%s: %s", owner, repo, exc
                    )
                    repo_counts["errors"] += 1
                    telemetry.log_event(
                        "error",
                        {
                            "owner": owner,
                            "repo": repo,
                            "issue_number": issue_payload.get("number"),
                            "error": str(exc),
                            "stage": "process_issue",
                        },
                    )
                    if args.debug:
                        raise
                    continue

            console_logger.info(
                "Repo summary %s/%s fetched=%s processed=%s skipped_labeled=%s "
                "skipped_low_confidence=%s errors=%s",
                owner,
                repo,
                repo_counts["fetched"],
                repo_counts["processed"],
                repo_counts["skipped_labeled"],
                repo_counts["skipped_low_confidence"],
                repo_counts["errors"],
            )
    except (RuntimeError, ValueError, OSError) as exc:
        if args.debug:
            raise
        console_logger.error("Agent failed: %s", exc)
        console_logger.error("Re-run with --debug to see the full traceback.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
