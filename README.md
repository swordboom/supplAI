# SupplAI - AI Supply Chain Disruption Monitor

SupplAI is an AI-assisted supply chain intelligence dashboard that turns disruption signals into operational decisions:

- parse disruption events (manual, news, weather, earthquakes)
- simulate downstream cascade impact on a supply network graph
- score node-level risk with ML + graph analytics
- recommend alternate routes with exposure validation
- explain ML risk drivers with SHAP
- detect anomalous shipment patterns
- generate executive briefs and autonomous agent action plans

## What The App Shows

The Streamlit dashboard currently exposes 7 tabs:

1. Network Graph
2. Risk Analysis
3. Rerouting
4. AI Brief
5. ML Explainability
6. Anomaly Detection
7. AI Agent

Core pipeline in code (`app.py`):

1. `parse_disruption(...)`
2. `run_cascade(...)`
3. `score_nodes(...)`
4. `find_alternates(...)`
5. `compute_shap(...)`
6. `generate_brief(...)`
7. `score_anomalies(...)`
8. `run_agent(...)`

## Quick Start (Local)

```bash
# 1) (Optional) activate your environment
conda activate condaVE

# 2) Install dependencies
pip install -r requirements.txt

# 3) Configure env vars (optional)
# Linux/macOS:
cp .env.example .env
# Windows PowerShell:
Copy-Item .env.example .env

# 4) Train/load local model artifacts + generate sanity report
python train_models.py

# 5) Run dashboard (inference app)
streamlit run app.py
```

Notes:

- `app.py` is inference-focused and loads artifacts from `models/`.
- `train_models.py` writes:
  - `models/model_sanity_report.json`
  - `evaluation.md`
- To force retraining:
  - `python train_models.py --force-retrain`

## Models Used

### 1) Delay prediction model (`models/delay_model.pkl`)

- Preferred runtime: `XGBoost` with CUDA when available
- Fallback runtime: `RandomForestClassifier` on CPU
- Used in:
  - composite risk scoring (`delay_prob` component)
  - SHAP explainability tab

### 2) Anomaly model (`models/anomaly_model.pkl`)

- Model: `IsolationForest` on city-level shipment features
- Used in:
  - anomaly detection tab
  - agent anomaly overlap checks

### 3) LLM providers (API-driven, not locally trained)

- Gemini (`gemini-2.5-flash`) for:
  - disruption parsing
  - live news event extraction
  - ops brief generation
  - optional agent reasoning enrichment
- Groq (`openai/gpt-oss-120b`) for:
  - brief fallback path
  - tool-calling autonomous agent path

## Works Without API Keys

SupplAI is designed to remain usable offline or key-less:

1. Disruption parsing falls back to deterministic keyword mapping.
2. News headlines still load from RSS; extraction falls back to keyword logic.
3. Earthquakes always come from USGS feed.
4. Weather falls back to Open-Meteo when OpenWeather key is absent.
5. AI brief falls back to template output.
6. AI agent falls back to deterministic decision flow.

## Optional Environment Variables

See `.env.example` for baseline keys:

- Gemini:
  - `GEMINI_API_KEY`
  - `GOOGLE_API_KEY`
  - `GOOGLE_GENAI_API_KEY`
- Groq:
  - `GROQ_API_KEY`
- OpenWeather:
  - `OPENWEATHER_API_KEY`
  - `OPENWEATHERMAP_API_KEY`
  - `OWM_API_KEY`

Cloud/Vertex-related vars used in code and deployment:

- `USE_VERTEX_AI`
- `GCP_PROJECT_ID`
- `GCP_REGION`
- `GCS_BUCKET_NAME`

## Training, Evaluation, And Optimization

### Baseline training + sanity checks

```bash
python train_models.py
python train_models.py --force-retrain
python train_models.py --skip-delay
python train_models.py --skip-anomaly
```

### Optional model optimization experiment

```bash
python optimize_models.py
```

Produces optimized artifacts/report in `models/` (for experimentation):

- `delay_model_optimized.pkl`
- `anomaly_model_optimized.pkl`
- `optimization_report.json`

## Data Generation Utilities

Regenerate synthetic/global dataset:

```bash
python generate_dataset.py
```

Regenerate pairwise tariff table:

```bash
python generate_tariffs.py
```

## Deployment

This repo includes:

- `Dockerfile` for containerized Streamlit serving
- `cloudbuild.yaml` for Google Cloud Build -> Artifact Registry -> Cloud Run

Cloud Build config also supports:

- Secret Manager injection for API keys
- Vertex AI mode (`USE_VERTEX_AI=true`)
- GCS-backed asset retrieval for models/data

## Project Structure

```text
supplAI/
|- app.py
|- train_models.py
|- optimize_models.py
|- generate_dataset.py
|- generate_tariffs.py
|- requirements.txt
|- .env.example
|- evaluation.md
|- Dockerfile
|- cloudbuild.yaml
|- data/
|- datasets/
|- models/
`- src/
```

## Security Note

- Never commit real credentials to source control.
- If any key is exposed, rotate it immediately at the provider console.
