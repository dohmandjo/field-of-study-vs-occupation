# CLAUDE.md

PySpark logistic-regression classifier that predicts whether someone is likely to change occupation, served by FastAPI and
queried from a React form. README.md has the user-facing overview, results and example request; this file is the working notes.

## Layout

- `ml/schema.py` - **single source of truth for column names.** Every feature has three names: `csv` (raw header), `model`
  (snake_case, inside Spark) and `api` (camelCase, JSON + React). Add/rename features here, never by hand elsewhere.
- `ml/pipeline.py` - StringIndexer -> OneHotEncoder -> VectorAssembler -> LogisticRegression as one `Pipeline`, plus `evaluate`,
  `explain`, `coefficients`.
- `ml/train.py` - CLI; writes `models/career_change_pipeline/`, `metrics.json`, `feature_means.json`.
- `api/main.py` - FastAPI `/predict` and `/health`; loads the saved `PipelineModel` once in the lifespan.
- `frontend/` - Vite + React 19 + TypeScript + Tailwind 3 form.
- `notebooks/` - 01 exploration, 02 modeling. Committed with outputs; excluded from ruff.
- `tests/` - pytest (schema, pipeline, API).
- `data/raw/field_of_study_vs_occupation.csv` is the only committed data. Everything under `models/`, `data/clean/`,
  `*.parquet`, `data/*.csv` is generated and git-ignored.

## Commands

Run from the repo root with the root `.venv` (Python 3.12 locally, 3.11 in CI). Needs **Java 17+** for PySpark.

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt          # runtime deps are in requirements.txt
pytest -q                                    # ~40 s; trains on a 3,000-row sample
ruff check .                                 # line length 120, rules E F I B
python -m ml.train --exclude careerChangeInterest
uvicorn api.main:app --port 8001
cd frontend && npm install && npm run dev    # npm run build = tsc -b + vite build (what CI runs)
```

CI (`.github/workflows/ci.yml`): ruff + pytest on Python 3.11/Java 17, and `npm ci && npm run build` for the frontend.

## Rules and gotchas

- **Training and serving share one fitted `PipelineModel`.** Never fit indexers/encoders in the API: the first version refitted
  StringIndexers per request and every category became index 0. Tests guard this (API and batch scoring must agree on a row;
  changing a category must change the prediction).
- **AUC comes from `rawPrediction`**, not the 0/1 `prediction` column.
- **`careerChangeInterest` leaks the label** (interest = 1 always means label = 1). Report metrics without it
  (0.870 AUC / 77.5 % accuracy vs a 57.7 % baseline). Don't quote the old 0.8945 / 0.897 figures.
- Unseen categories don't fail: `handleInvalid="keep"` scores them with no signal, and the API lists them in
  `unseen_categories`.
- `PredictionRequest` uses `extra="forbid"` and range checks; invalid input returns 422. Keep the API fields, `ml/schema.py`
  and the React form in sync.
- Tests: the `spark` fixture is function-scoped on purpose, because the API lifespan stops its Spark session.
- `requirements.txt` adds `setuptools` on Python 3.12+ because pyspark 3.5 still imports `distutils`.
- Config: `MODEL_DIR` (default `./models`) and `CORS_ORIGINS` (default `http://localhost:5173`) for the API; `VITE_API_URL`
  (default `http://localhost:8001`) in `frontend/.env`.
- The dataset looks synthetic (categories almost perfectly balanced), so frame results as a demo of the pipeline.
- The old Django scaffold, `ui/` app and root `package.json` were removed on purpose. Don't recreate them; the stack is
  PySpark ML + FastAPI + React.
