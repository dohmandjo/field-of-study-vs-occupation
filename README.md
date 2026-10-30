# Field of Study vs Occupation - career-change prediction

[![CI](https://github.com/dohmandjo/field-of-study-vs-occupation/actions/workflows/ci.yml/badge.svg)](https://github.com/dohmandjo/field-of-study-vs-occupation/actions/workflows/ci.yml)

A small end-to-end machine-learning project: explore a career dataset with **PySpark**, train a **logistic regression** pipeline,
serve it with **FastAPI** and query it from a **React + Tailwind** form.

```
data/raw/*.csv ──► ml/ (PySpark pipeline, train CLI) ──► models/career_change_pipeline
                                                             │
        React form (frontend/) ──HTTP──► FastAPI (api/) ─────┘
```

## Results (single 80/20 split, seed 42, 7,675 test rows)

| Model | Accuracy | AUC | vs. majority baseline (57.7 %) |
|---|---|---|---|
| All 22 features | 90.2 % | 0.966 | +32.6 pts |
| **Without `Career Change Interest`** | 77.5 % | 0.870 | +19.8 pts |

**Read the first row with care.** In this dataset every person with `Career Change Interest = 1` (7,685 of 38,444) is labelled
*likely to change*, so that one column carries most of the score (its coefficient is about 25x the next feature). The second row is
the honest estimate if you want to predict change *before* someone says they are interested. The data also looks synthetic (every
category is almost perfectly balanced), so these numbers demonstrate the pipeline, not real-world career behaviour.
Details and plots: [`notebooks/02_feature_engineering_and_modeling.ipynb`](notebooks/02_feature_engineering_and_modeling.ipynb).

## Repository layout

```
.
├── data/raw/                  source CSV (38,444 rows, 22 features + label)
├── ml/
│   ├── schema.py              single source of truth for column names (CSV / Spark / API)
│   ├── pipeline.py            StringIndexer -> OneHotEncoder -> VectorAssembler -> LogisticRegression, evaluation, explanations
│   └── train.py               CLI: trains, evaluates, saves models/
├── api/main.py                FastAPI service: /predict, /health
├── frontend/                  Vite + React 19 + Tailwind form
├── notebooks/                 01 exploration, 02 modelling (outputs included)
├── tests/                     pytest: schema, pipeline, API (23 tests)
└── .github/workflows/ci.yml   ruff + pytest (Java 17) + frontend build
```

## Run it

Requirements: Python 3.11+, **Java 17+** (PySpark), Node 20+ for the frontend.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. train (writes models/, which is git-ignored)
python -m ml.train                                   # all features
python -m ml.train --exclude careerChangeInterest    # without the leaky feature (recommended)

# 2. serve
uvicorn api.main:app --port 8001

# 3. frontend
cd frontend && cp .env.example .env && npm install && npm run dev      # http://localhost:5173
```

Example request:

```bash
curl -X POST localhost:8001/predict -H 'content-type: application/json' -d '{
  "fieldOfStudy":"Medicine","currentOccupation":"Doctor","gender":"Male","educationLevel":"PhD",
  "industryGrowthRate":"Low","familyInfluence":"None","age":40,"yearsOfExperience":10,"jobSatisfaction":2,
  "workLifeBalance":3,"jobOpportunities":20,"salary":90000,"jobSecurity":4,"careerChangeInterest":0,"skillsGap":5,
  "mentorshipAvailable":0,"certifications":0,"freelancingExperience":1,"geographicMobility":1,
  "professionalNetworks":5,"careerChangeEvents":1,"technologyAdoption":5}'
```

The response holds `prediction` (0/1), `probability` (of *likely to change*), `probability_0`, `top_features` with their signed
`effects` on the log-odds relative to an average training row, and `unseen_categories` (categories the model never saw; they are
scored with no signal instead of failing). Invalid input (age 5, satisfaction 11 ...) returns HTTP 422.

Configuration: `MODEL_DIR` (default `./models`) and `CORS_ORIGINS` (default `http://localhost:5173`) for the API,
`VITE_API_URL` (default `http://localhost:8001`) for the frontend.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # trains on a 3,000-row sample; about one minute
ruff check .
```

The suite includes regression tests for the bugs of the first version: API and batch scoring must agree on the same row, changing a
category must change the prediction, and AUC must come from `rawPrediction`.

## Limitations

- One random split, no cross-validation or tuning; a logistic regression is the only model.
- The API scores with Spark in local mode: correct and consistent with training, but heavier than needed for a single row.
- The model is trained on ages 20-59 and 1-10 rating scales; the API accepts wider ranges and does not warn about extrapolation.
