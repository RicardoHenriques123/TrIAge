
# Technical Implementation Report: AI-Powered Support Triage & Automation Pipeline

**Document Status:** Draft
**Author:** Ricardo Henriques
**Project:** GitHub Issue Operations Agent (Hybrid LLM Architecture)

---

## 1. Executive Summary
This document outlines the architectural design and technical implementation strategy for an automated support triage system. The system leverages a Hybrid Routing Architecture, combining a fast, lightweight classical Machine Learning classifier with a Large Language Model fallback. This approach ensures high-speed, cost-efficient resolution for standard operational requests while retaining the generative reasoning capabilities of an LLM for complex, unstructured, or ambiguous issues.

## 2. High-Level System Architecture
The system is bifurcated into two distinct operational environments to separate the compute-heavy model training from the lightweight, idempotent production agent.

### Phase 1: Offline Training Sandbox
This environment is dedicated to the extraction, transformation, and modeling of historical operational data. Implemented in `triage/training/`, this phase is triggered via `scripts/train_model.py`.
* **Data Acquisition Engine:** Utilizes the GitHub REST API to fetch closed, labeled issues from specified target repositories. Supports label allowlisting to focus on specific operational categories, creating a ground-truth dataset.
* **Preprocessing Pipeline:** Implements text sanitization (removing markdown, code blocks, and artifacts) via `triage.preprocessing.text.sanitize_text()`. Vectorization uses TF-IDF with unigram and bigram features, English stop-word removal, and a configurable feature limit.
* **Model Training & Evaluation:** Trains a TF-IDF + Logistic Regression pipeline (`sklearn.pipeline.Pipeline`) on the vectorized dataset. Optionally supports train/test evaluation with stratified splitting and detailed classification reports for model validation.
* **Model Serialization:** The trained pipeline is persisted as a `.joblib` file for production use. Accompanying metadata (labels, sample counts, label distribution) is saved as a `.meta.json` file for runtime reference.

### Phase 2: Online Inference Engine (Live Agent)
A stateless, event-driven script implemented in `triage/agent/run.py` and triggered via `scripts/run_agent.py`. Designed to run on a cron job or webhook trigger.
* **State Management:** Designed for idempotency. The agent fetches only "open" issues from the GitHub API and explicitly skips any issues that already have labels, ensuring no duplicate processing or redundant updates.
* **Hybrid Router:** Implemented via the Chain of Responsibility pattern in `triage/routing/handlers.py`. Raw issue text is sanitized and passed through the deserialized local ML model. 
    * *Path A (High Confidence):* The `LocalModelHandler` calls the model's `predict()` method and extracts the confidence score. If confidence $\geq$ the configured threshold (default 0.80), it returns `handled=True` with the predicted label.
    * *Path B (Low Confidence Fallback):* If the local model's confidence falls below the threshold or the model is unavailable, the `LLMHandler` (if configured) receives the payload. The `LLMHandler` invokes the OpenRouter API for semantic analysis and categorization, returning `handled=True` with either a predicted label or error status.
    * *Path C (Unhandled):* If no LLM is configured and the local model is low-confidence, the issue is skipped and logged as unresolved (requires `--allow-no-llm` flag).
* **Action Execution:** For handled issues, the agent interfaces with the GitHub API to apply labels via `adapter.add_labels()`. Optionally, it posts a comment (either LLM-generated or an auto-generated confidence message). Both operations are skipped in dry-run mode (`--dry-run` flag).
* **Telemetry & Logging:** Every routing decision is logged with full decision context (confidence, source, local vs. LLM prediction, duration) via `TelemetryLogger` to structured JSON format, enabling dashboard integration and operational analysis.

## 3. Software Engineering Design Patterns
To ensure the system remains maintainable, scalable, and adheres to SOLID principles, the following design patterns are implemented within the Online Inference Engine:

* **Chain of Responsibility Pattern (The Core Router):** Manages the Hybrid Routing logic. The payload is passed to the `LocalModelHandler`. If the internal confidence score evaluates to `False`, the handler seamlessly passes the payload to the next node in the chain, the `LLMHandler`, preventing complex and brittle nested conditional logic.
* **Strategy Pattern (Classifier Abstraction):** Defines a unified `predict()` interface through the `ClassifierStrategy` protocol. Both the `scikit-learn` models and the `OpenRouter` API wrappers implement this interface. This allows operations teams to swap underlying classification models without refactoring the core business logic.
* **Adapter Pattern (API Gateways):** Isolates external dependencies. Dedicated `GitHubAdapter` and `OpenRouterAdapter` classes wrap the respective third-party SDKs and REST calls. This centralizes rate-limit handling, pagination, and authentication.

## 4. Telemetry and Operational Monitoring
Visibility into the agent's performance is critical for continuous operational improvement.
* **Logging:** Every routing decision, latency metric, and confidence score is logged to a structured format (CSV/JSON).
* **Dashboard Integration:** The structured logs are formatted to be seamlessly ingested by **Google Looker Studio**. This enables the visualization of automated resolution rates, model fallback frequencies, and category distribution, providing actionable insights into operational bottlenecks.

## 5. Implementation Structure
The codebase is organized around the core `triage/` package:

* **`triage/adapters/`:** Contains API client wrappers (`github.py`, `openrouter.py`) that isolate external dependencies and centralize rate-limiting, pagination, and authentication.
* **`triage/agent/`:** Contains `run.py`, the main Online Inference Engine entry point that orchestrates the routing pipeline.
* **`triage/models/`:** Contains the classifier abstractions (`base.py`, `llm.py`, `local_model.py`) implementing the Strategy pattern for interchangeable models.
* **`triage/preprocessing/`:** Contains `text.py` for issue text sanitization and normalization.
* **`triage/routing/`:** Contains `handlers.py` implementing the Chain of Responsibility pattern with `BaseHandler`, `LocalModelHandler`, and `LLMHandler`.
* **`triage/telemetry/`:** Contains `logger.py` for structured event logging and monitoring.
* **`triage/training/`:** Contains `dataset.py` and `train.py` for the Offline Training Sandbox phase.

## 6. Security & Infrastructure Considerations
* **Secrets Management:** Environment variables (`.env`) are strictly utilized for GitHub Personal Access Tokens (PAT) and OpenRouter API keys.
* **Extensibility:** The modular design allows the ingestion engine to be easily adapted from GitHub Issues to internal ticketing systems (e.g., Jira, Zendesk) by implementing a new Adapter class.

