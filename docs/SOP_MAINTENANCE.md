# SOP Maintenance & Extensibility Guide

This document provides technical procedures for maintaining the triage system, monitoring its performance, troubleshooting issues, and extending it to support new data sources and classification strategies.

For initial setup and quick-start instructions, refer to [README.md](../README.md#-quick-start).

**System Summary**
- Offline training pipeline (`triage/training/`) builds a local TF-IDF + Logistic Regression classifier from historical labeled GitHub issues.
- Online inference agent (`triage/agent/`) routes issues through a local model and optionally an LLM fallback via the Chain of Responsibility pattern.
- Telemetry is written as JSON Lines to `logs/triage.jsonl` for operational reporting and dashboard integration.

---

## Prerequisites & Environment
- Python 3.10+
- GitHub Personal Access Token (PAT) with repo read/write access
- OpenRouter API key (optional but required for LLM fallback)

Environment configuration is loaded from `.env` via `triage/config.py`. See README.md for the full list of configuration variables.

---

## Telemetry & Operational Monitoring

### Telemetry Schema
All routing decisions are logged as JSON objects to `logs/triage.jsonl`. Each decision record contains:

**Core Decision Fields**
- `event`: Event type (`"decision"` or `"error"`)
- `label`: Applied label (or empty if skipped)
- `confidence`: Model confidence score (local or LLM)
- `source`: Classification source (`"local"` or `"llm"`)
- `handled`: Whether the issue was processed

**Local Model Visibility**
- `local_label`: Prediction from the local model
- `local_confidence`: Confidence score from the local model (0.0 if unavailable)
- `skipped`: Boolean indicating if the issue was skipped (low confidence, no fallback)
- `reason`: If skipped, the reason (e.g., `"low_confidence_no_llm"`)

**Performance & Context**
- `duration_seconds`: Latency in seconds (rounded to 4 decimals)
- `owner`, `repo`, `issue_number`, `issue_url`: GitHub context
- `dry_run`: Whether the run was in dry-run mode

**Error Records**
If `event="error"`, additional fields appear:
- `stage`: Where the error occurred (e.g., `"list_labels"`, `"iteration"`)
- `error`: Error message

### Telemetry Analysis

**Query: Local Model Coverage**
```python
import json
handled_local = 0
handled_llm = 0
skipped = 0
with open("logs/triage.jsonl") as f:
    for line in f:
        record = json.loads(line)
        if not record.get("handled"):
            skipped += 1
        elif record.get("source") == "local":
            handled_local += 1
        else:
            handled_llm += 1
print(f"Local: {handled_local}, LLM: {handled_llm}, Skipped: {skipped}")
```

**Operational Checks**
- If `local_confidence` is always `0.0`, the model file may be missing or corrupted. Verify `models/issue_classifier.joblib` exists.
- If `source="llm_error"` appears frequently, check OpenRouter service status and increase `OPENROUTER_TIMEOUT` in `.env`.
- High `skipped` rate indicates the threshold is too high; consider lowering `--threshold` or retraining with more domain-relevant data.
- Track `duration_seconds` over time. Spikes suggest OpenRouter latency or network issues.

---

## Troubleshooting

### Common Issues

**OpenRouter Timeout**
```
Error: Request to OpenRouter timed out
```
**Solution:** Increase `OPENROUTER_TIMEOUT` in `.env` (default 45 seconds).
```env
OPENROUTER_TIMEOUT=60
```

**Invalid Model ID**
```
Error: Model openai/gpt-4x not found
```
**Solution:** Set a valid `OPENROUTER_MODEL` in `.env`. Valid examples: `openai/gpt-4o-mini`, `anthropic/claude-3-haiku`.

**Local Model Never Triggered**
Symptom: All issues routed to LLM (`source="llm"` in telemetry).

**Solution:** Lower `--threshold` or retrain the model with more domain-matched data.
```bash
# Try a more conservative threshold
python scripts/run_agent.py --repos owner/repo --threshold 0.70

# Or retrain with more targeted data
python scripts/train_model.py --repos owner/repo --label-allowlist bug feature --eval
```

**Model File Corruption**
```
Error: Failed to load local model
```
**Solution:** Retrain the model:
```bash
python scripts/train_model.py --repos owner/repo --output models/issue_classifier.joblib
```

**GitHub API Rate Limiting**
Symptom: Requests fail with 403 status.

**Solution:** Use a fresh GitHub token with sufficient rate limit or implement request batching.

### Debugging

Use `--debug` flag to enable full stack traces and fail-fast behavior:
```bash
python scripts/train_model.py --repos owner/repo --debug
python scripts/run_agent.py --repos owner/repo --debug
```

---

## Maintenance Schedule

### Daily
- Monitor `logs/triage.jsonl` for errors or unexpected patterns.
- Check GitHub Actions logs if the agent runs on a scheduled trigger.

### Weekly
- Review telemetry for high LLM fallback rate (indicates threshold may be too high).
- Monitor average `duration_seconds` for performance degradation.
- Check OpenRouter API usage and costs.

### Monthly
- Retrain the model if label distribution or issue types shift.
- Rotate GitHub tokens if required by policy.
- Re-evaluate label taxonomy and update `--label-allowlist`.
- Review and update confidence threshold based on precision/recall tradeoffs observed in telemetry.

### Quarterly
- Full model evaluation: Retrain with `--eval` and review classification report.
- Audit OpenRouter model choice; newer models may offer better accuracy or cost.
- Review security posture: rotate all credentials, audit token permissions.

---

## Model Refresh & Retraining

### When to Retrain
- New label types are needed
- Issue vocabulary or patterns change significantly
- Classification accuracy degrades (observed via telemetry)
- Model file is corrupted or missing

### Retraining Procedure

1. **Prepare training data:**
   ```bash
   python scripts/train_model.py \
     --repos owner/repo1 owner/repo2 \
     --label-allowlist bug feature enhancement documentation \
     --max-issues 1000 \
     --eval
   ```

2. **Review evaluation metrics:**
   - Check the printed classification report (precision, recall, F1-score).
   - Verify that `models/issue_classifier.meta.json` contains all expected labels.

3. **Validate in dry-run mode:**
   ```bash
   python scripts/run_agent.py \
     --repos owner/repo1 owner/repo2 \
     --dry-run \
     --threshold 0.80
   ```
   Review telemetry to ensure routing decisions look reasonable.

4. **Deploy to production:**
   Replace the model file on the inference host(s) and resume live runs.

---

## Security SOP

**Secrets Management**
- Never commit `.env` or credentials to version control.
- Use `.gitignore` to exclude `.env` and credential files.
- Rotate credentials on a regular schedule (monthly recommended).

**GitHub Token Best Practices**
- Create tokens with minimal required scopes (e.g., `repo:read`, `pull_request:read` only if the agent does not modify repos).
- Use organization-level tokens if available to centralize access control.
- Revoke tokens immediately if compromised.

**OpenRouter API Key Best Practices**
- Store keys in environment variables or secure vaults, never in code.
- Use rate-limiting and budget alerts in the OpenRouter dashboard.
- Rotate keys quarterly.

**Code Review & Audit**
- Review changes to `triage/adapters/` (API interactions) and `triage/routing/` (decision logic) carefully.
- Audit telemetry logs for unexpected API calls or classification patterns.

---

## Extensibility Guide

The modular architecture enables extension without modifying core logic. This section describes how to implement custom adapters, classifiers, and routing strategies.

### Implementing a Custom Adapter

Adapters isolate external dependencies (APIs, databases, filesystems). To add support for a new ticketing system (e.g., Jira, Zendesk):

1. **Create a new adapter module** in `triage/adapters/`:
   ```python
   # triage/adapters/jira.py
   """Jira adapter for issue triage."""
   from typing import Dict, Any, List, Tuple
   import requests
   
   class JiraAdapter:
       """Adapter for Jira Cloud API."""
       
       def __init__(self, token: str, base_url: str):
           self.token = token
           self.base_url = base_url
       
       def iter_issues(self, project: str, state: str = "open", max_issues: int = None):
           """Iterate over issues. Signature matches GitHubAdapter for consistency."""
           # Implement Jira API pagination
           pass
       
       def add_labels(self, issue_id: str, labels: List[str]):
           """Apply labels to an issue."""
           # Implement Jira label API update
           pass
       
       def create_comment(self, issue_id: str, body: str):
           """Post a comment on an issue."""
           # Implement Jira comment API
           pass
   ```

2. **Maintain interface consistency:**
   - Follow the same method signatures as `GitHubAdapter` (`iter_issues`, `add_labels`, `create_comment`).
   - Ensure pagination and error handling are robust.

3. **Update the agent entrypoint:**
   Modify `triage/agent/run.py` to conditionally instantiate the adapter based on configuration:
   ```python
   if config.adapter_type == "github":
       adapter = GitHubAdapter(...)
   elif config.adapter_type == "jira":
       adapter = JiraAdapter(...)
   ```

### Implementing a Custom Classifier

Classifiers predict labels for issue text. To swap the default Logistic Regression for a different model:

1. **Implement the `ClassifierStrategy` protocol** in `triage/models/`:
   ```python
   # triage/models/custom_model.py
   """Custom classifier implementation."""
   from typing import List, Optional
   from triage.models.base import ClassificationResult
   
   class CustomClassifier:
       """Custom ML classifier (e.g., XGBoost, neural network)."""
       
       def __init__(self, model_path: str):
           # Load your custom model
           self.model = self._load_model(model_path)
       
       def predict(self, text: str, labels: Optional[List[str]] = None) -> ClassificationResult:
           """Predict a label for the given text."""
           # Your prediction logic
           label = self.model.predict(text)
           confidence = self.model.confidence(text)  # Custom method
           return ClassificationResult(label=str(label), confidence=float(confidence), source="custom")
   ```

2. **Update the training pipeline** to train your model:
   ```python
   # In triage/training/train.py or a new module
   def train_custom_model(texts: List[str], labels: List[str], output_path: Path):
       # Your training logic (e.g., XGBoost, PyTorch, etc.)
       pass
   ```

3. **Register the classifier in the agent:**
   Modify `triage/agent/run.py`:
   ```python
   if config.classifier_type == "logistic_regression":
       classifier = LocalModelClassifier(model_path)
   elif config.classifier_type == "custom":
       classifier = CustomClassifier(model_path)
   ```

### Extending the Routing Chain

The Chain of Responsibility pattern allows insertion of custom handlers between `LocalModelHandler` and `LLMHandler`.

1. **Create a custom handler** in `triage/routing/`:
   ```python
   # triage/routing/custom_handlers.py
   """Custom routing handlers."""
   from triage.routing.handlers import BaseHandler, HandlerResult
   from triage.models.base import ClassificationResult
   
   class RuleBasedHandler(BaseHandler):
       """Handler that applies deterministic rules before ML."""
       
       def __init__(self):
           super().__init__()
           self.rules = {
               "security": ["vulnerability", "CVE", "buffer overflow"],
               "documentation": ["docs", "readme", "wiki"],
           }
       
       def _handle(self, text: str, labels: Optional[List[str]] = None) -> HandlerResult:
           # Check if text matches any rule
           for label, keywords in self.rules.items():
               if any(kw.lower() in text.lower() for kw in keywords):
                   result = ClassificationResult(label=label, confidence=1.0, source="rule")
                   return HandlerResult(handled=True, result=result)
           # Pass to next handler
           return HandlerResult(handled=False, result=None)
   ```

2. **Insert the handler into the chain:**
   Modify `triage/agent/run.py`:
   ```python
   def build_chain(model_path, threshold, llm_classifier):
       rule_handler = RuleBasedHandler()
       local_handler = LocalModelHandler(LocalModelClassifier(model_path), threshold)
       
       rule_handler.set_next(local_handler)
       if llm_classifier:
           local_handler.set_next(LLMHandler(llm_classifier))
       
       return rule_handler  # Return the first handler
   ```

### Custom Configuration

To support new configuration options:

1. **Add fields to `triage/config.py`:**
   ```python
   @dataclass(frozen=True)
   class Config:
       # ... existing fields ...
       adapter_type: str  # "github" or "jira"
       classifier_type: str  # "logistic_regression" or "custom"
   ```

2. **Set defaults and load from environment:**
   ```python
   DEFAULT_ADAPTER_TYPE = "github"
   DEFAULT_CLASSIFIER_TYPE = "logistic_regression"
   
   config = Config(
       adapter_type=os.getenv("ADAPTER_TYPE", DEFAULT_ADAPTER_TYPE),
       classifier_type=os.getenv("CLASSIFIER_TYPE", DEFAULT_CLASSIFIER_TYPE),
       # ... other fields ...
   )
   ```

---

## Performance Optimization

### Model Size & Inference Speed
The TF-IDF + Logistic Regression pipeline is designed to be lightweight. Typical inference latency is < 100ms per issue.

To optimize further:
- Reduce TF-IDF `max_features` (currently 5000) in `triage/training/train.py`
- Use sparse matrix representations in custom classifiers
- Batch requests to the LLM fallback if latency becomes a bottleneck

### Caching Strategy
Consider implementing a caching layer for frequently-seen issues:
```python
# In LocalModelHandler or between handlers
class CachingHandler(BaseHandler):
    def __init__(self, delegate: BaseHandler):
        super().__init__()
        self.cache = {}  # text -> ClassificationResult
        self.delegate = delegate
    
    def handle(self, text: str, labels: Optional[List[str]] = None) -> HandlerResult:
        if text in self.cache:
            return HandlerResult(handled=True, result=self.cache[text])
        result = self.delegate.handle(text, labels)
        self.cache[text] = result.result
        return result
```

---

## 🧪 Testing

The project includes a comprehensive test suite for unit, integration, and acceptance testing. 

### Quick Test Commands

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
make test

# Run with coverage report
make test-cov-html

# Run specific test category
make test-unit        # Unit tests only
make test-integration # Integration tests only
```

**Test Structure:**
- Unit tests for individual components (preprocessing, models, routing, telemetry)
- Integration tests for end-to-end pipelines
- Mock fixtures for external APIs (GitHub, OpenRouter)
- Coverage targets: 90%+ for core modules

## References
- [TECHNICAL_SPEC.md](./TECHNICAL_SPEC.md): Detailed architecture and design patterns
- [README.md](../README.md): Quick-start guide and project overview
- `triage/` package: Implementation details
