# SupplAI - AI Supply Chain Disruption Monitor

Hackathon project for disruption simulation, risk scoring, reroute planning, anomaly detection, and ops briefing.

## Quick Start

```bash
# 1) Activate environment
conda activate condaVE

# 2) Install dependencies
pip install -r requirements.txt

# 3) Configure environment variables (optional)
cp .env.example .env

# 4) Train local models + generate sanity report
python train_models.py

# 5) Run dashboard (inference only)
streamlit run app.py
```

Notes:
- `.env` is already gitignored.
- You can run the app without any API keys.
- `app.py` now loads pretrained artifacts from `models/` and does not train models.
- `train_models.py` writes a sanity report to `models/model_sanity_report.json`.
- `train_models.py` also writes a human-readable snapshot to `evaluation.md`.

## Models Used

1. Delay prediction model:
- Preferred: `XGBoost Classifier` (`xgboost_cuda`) when CUDA is available.
- Fallback: `RandomForestClassifier` (`random_forest_cpu`) when XGBoost/CUDA is unavailable.
- Saved artifact: `models/delay_model.pkl`
- Used by:
  - risk scoring delay-probability component
  - SHAP explainability tab

2. Anomaly detection model:
- `IsolationForest` on city-level shipment features.
- Saved artifact: `models/anomaly_model.pkl`
- Used by:
  - anomaly detection tab
  - agent anomaly alerts

3. LLM models (API-driven, not trained locally):
- `gemini-2.5-flash` (news extraction, event parsing, brief generation, agent reasoning enrichment)
- `llama-3.3-70b-versatile` via Groq (brief + agent tool-calling fallback path when `GROQ_API_KEY` is present)

## Inference-Only App Behavior

- `app.py` only loads model artifacts.
- If `models/delay_model.pkl` is missing, the app falls back to rule-based delay proxies and shows a warning.
- If `models/anomaly_model.pkl` is missing, anomaly outputs are skipped and the app shows a warning.
- To retrain artifacts, run:
  - `python train_models.py`
  - `python train_models.py --force-retrain` (forces full retrain)

## What Works Without API Keys

These features work fully or with built-in fallback logic even when `.env` is empty:

1. Disruption parsing and cascade/risk/reroute pipeline:
- Uses deterministic keyword parser fallback (`src/disruption_input.py`) when Gemini is unavailable.

2. Live news ingestion:
- Headlines are still fetched from public RSS feeds (`src/news_fetcher.py`).
- If Gemini key is missing, events are extracted by keyword-based fallback from headlines.

3. Live weather + earthquake monitoring:
- Earthquakes always come from USGS public feed.
- If OpenWeather key is missing, weather checks automatically use Open-Meteo public API fallback.

4. AI operations brief:
- If no AI provider key works, app generates a structured template brief (`src/llm_brief.py`).

5. Autonomous AI agent tab:
- If no Groq key is available, deterministic agent flow runs and still produces an action plan.
- If Gemini exists, deterministic results can be enriched with Gemini reasoning.

## Optional API Keys

You only need keys for enhanced LLM/weather behavior.

- Gemini / Google GenAI (any one):
  - `GEMINI_API_KEY`
  - `GOOGLE_API_KEY`
  - `GOOGLE_GENAI_API_KEY`
- Groq:
  - `GROQ_API_KEY`
- OpenWeather (any one):
  - `OPENWEATHER_API_KEY`
  - `OPENWEATHERMAP_API_KEY`
  - `OWM_API_KEY`

See `.env.example` for the full template.

## Security Note

- Never commit real credentials to source control.
- If a key was ever exposed in docs/history, rotate it immediately in the provider console.

## Project Structure

```text
supplAI/
|- app.py
|- train_models.py
|- requirements.txt
|- .env.example
|- data/
|- datasets/
|- models/
`- src/
```
