"""Training pipeline for the local issue classifier."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from triage.adapters.github import GitHubAdapter
from triage.config import load_config
from triage.training.dataset import build_dataset, summarize_labels


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configure a console logger for training.

    Args:
        verbose: Whether to enable debug logging.

    Returns:
        Configured logger.
    """

    logger = logging.getLogger("triage.training")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
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


def train_model(
    texts: List[str],
    labels: List[str],
    output_path: Path,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    logger: Optional[logging.Logger] = None,
) -> Pipeline:
    """Train and persist a TF-IDF + Logistic Regression model.

    Args:
        texts: Training texts.
        labels: Training labels.
        output_path: Model output path.
        max_features: TF-IDF max features.
        ngram_range: TF-IDF n-gram range.
        logger: Optional logger.

    Returns:
        Trained sklearn Pipeline.
    """

    if logger:
        logger.info("Training model with %s samples", len(texts))
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
    )
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=1,
    )
    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", classifier),
        ]
    )
    pipeline.fit(texts, labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    if logger:
        logger.info("Saved model to %s", output_path)
    return pipeline


def main() -> None:
    """CLI entrypoint for training the local classifier."""

    parser = argparse.ArgumentParser(description="Train local ML model for issue triage")
    parser.add_argument("--repos", nargs="+", required=True, help="owner/repo list")
    parser.add_argument("--output", default=None, help="Output model path")
    parser.add_argument("--label-allowlist", nargs="*", default=None)
    parser.add_argument("--max-issues", type=int, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logger = setup_logger(verbose=args.verbose)
    start = time.time()

    try:
        config = load_config()
        adapter = GitHubAdapter(token=config.github_token, api_url=config.github_api_url)
        repos = parse_repos(args.repos)
        logger.info("Starting training dataset build for %s repo(s)", len(repos))

        texts, labels = build_dataset(
            adapter=adapter,
            repos=repos,
            label_allowlist=args.label_allowlist,
            max_issues_per_repo=args.max_issues,
            progress_cb=lambda msg: logger.info(msg),
            fail_fast=args.debug,
        )

        if not texts:
            raise RuntimeError("No labeled issues found for training.")

        output_path = Path(args.output) if args.output else config.model_path
        logger.info(
            "Collected %s samples across %s labels", len(texts), len(set(labels))
        )
        logger.info("Label distribution: %s", summarize_labels(labels))

        if args.eval and len(texts) > 5:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            logger.info("Running train/test evaluation split")
            pipeline = train_model(
                X_train, y_train, output_path=output_path, logger=logger
            )
            predictions = pipeline.predict(X_test)
            report = classification_report(y_test, predictions, zero_division=0)
            logger.info("Evaluation report:\n%s", report)
        else:
            pipeline = train_model(texts, labels, output_path=output_path, logger=logger)

        meta = {
            "labels": sorted(set(labels)),
            "samples": len(labels),
            "label_counts": summarize_labels(labels),
            "model_path": str(output_path),
        }
        meta_path = output_path.with_suffix(".meta.json")
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=True)
        logger.info("Saved metadata to %s", meta_path)
        logger.info("Training complete in %.2fs", time.time() - start)
    except (RuntimeError, ValueError, OSError) as exc:
        if args.debug:
            raise
        logger.error("Training failed: %s", exc)
        logger.error("Re-run with --debug to see the full traceback.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
