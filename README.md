# 🚀 TrIAge - AI-Powered Operations & Triage Agent

TrIAge is an intelligent automation pipeline designed to modernize departmental workflows by bridging the gap between high-volume manual requests and automated resolution. 


## 🌐 Overview
In high-growth environments manual bottlenecks in triage can slow down technical teams. This project implements a **Hybrid AI Triage System** that:
* **Identifies & Categorizes:** Automatically sorts incoming GitHub issues/tickets using Machine Learning.
* **Optimizes Costs:** Uses a lightweight local model for standard requests and reserves advanced LLMs (via OpenRouter) for complex reasoning.
* **Automates Action:** Executes real-world operations by auto-labeling issues and drafting contextual responses to users.

---

## 🛠️ Key Features
* **Hybrid Routing Engine:** A dual-layer classification system that balances speed, cost, and accuracy.
* **Production API Integration:** Fully integrated with the GitHub REST API and OpenRouter.
* **Data-Driven Insights:** Generates operational telemetry formatted for visualization in **Google Looker Studio** or **Tableau**.
* **Enterprise Standards:** Built with strict adherence to SOLID principles to allow seamless extension, including for other internal ticketing systems.

---

## ⚡ Quick Start

### 1. Installation & Configuration
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with the following credentials:
```env
GITHUB_TOKEN=your_github_personal_access_token
OPENROUTER_API_KEY=your_openrouter_api_key
```

Optional environment variables:
```env
OPENROUTER_MODEL=openai/gpt-4o-mini          # Default LLM model
OPENROUTER_TIMEOUT=45                        # Request timeout (seconds)
CONFIDENCE_THRESHOLD=0.80                    # Local ML threshold
MODEL_PATH=models/issue_classifier.joblib    # Serialized model path
LOG_PATH=logs/triage.jsonl                   # Telemetry output
FALLBACK_LABEL=needs-triage                  # Default label for ambiguous issues
DRY_RUN=false                                # Safety flag (true = no GitHub mutations)
```

### 2. Train the Local Model
Before running the agent, train a classifier on historical closed issues from target repositories:
```bash
python scripts/train_model.py \
  --repos owner/repo1 owner/repo2 \
  --label-allowlist bug feature enhancement \
  --eval
```

This will:
- Fetch closed issues from the specified repositories
- Filter for the whitelisted labels
- Train a TF-IDF + Logistic Regression model
- Export `models/issue_classifier.joblib` and `models/issue_classifier.meta.json`

For all available options, run: `python scripts/train_model.py --help`

### 3. Run the Triage Agent
Once trained, deploy the agent to triage open issues:
```bash
python scripts/run_agent.py \
  --repos owner/repo1 owner/repo2 \
  --dry-run
```

**Recommended workflow for first deployment:**
1. Start with `--dry-run` to preview decisions without applying changes
2. Use `--threshold 0.85` to be more conservative initially
3. Use `--comment-on-high-confidence` to post explanatory comments
4. Monitor logs at `logs/triage.jsonl` to validate decisions

**Example - Conservative Dry-Run:**
```bash
python scripts/run_agent.py \
  --repos owner/repo1 owner/repo2 \
  --dry-run \
  --threshold 0.85 \
  --comment-on-high-confidence
```

**Example - Production Run with Full Fallback:**
```bash
python scripts/run_agent.py \
  --repos owner/repo1 owner/repo2
```

For all available options, run: `python scripts/run_agent.py --help`  
For troubleshooting and advanced configuration, see [SOP_MAINTENANCE.md](./docs/SOP_MAINTENANCE.md).

### 4. Monitor Outputs
- **Telemetry Logs:** Check `logs/triage.jsonl` for decision history (importable to Looker Studio or Tableau)
- **Console Output:** Real-time routing decisions and error logs

---

## 📂 Documentation Hierarchy
To ensure sustainability and ease of maintenance, this project is documented across three levels:

1.  **README.md (This File):** Project vision, value proposition, and quick-start.
2.  [**TECHNICAL_SPEC.md**](./docs/TECHNICAL_SPEC.md): Deep dive into the system architecture, design patterns, and ML methodology.
3.  [**SOP_MAINTENANCE.md**](./docs/SOP_MAINTENANCE.md): Standard Operating Procedures for developers and ops teams (setup, model rotation, troubleshooting, testing and extensibility).


## 🔮 Future Roadmap: AutoMLOps Integration
While the current architecture successfully bridges the gap between manual triage and automated routing, the next phase of development focuses on establishing a self-optimizing ML pipeline.

* **Autonomous Model Research (RD-Agent):** I plan to integrate Microsoft's **RD-Agent** to act as an autonomous Data Scientist within the training environment. By decoupling the data ingestion from the training loop, RD-Agent will be able to iteratively experiment with `train_model.py`—autonomously testing different feature engineering techniques, n-gram ranges, and classifier algorithms (e.g., LightGBM, Random Forest)—to continuously maximize the F1-score of the local routing model without manual human intervention.