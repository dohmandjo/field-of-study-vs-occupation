import pytest
from fastapi.testclient import TestClient
from pyspark.ml import PipelineModel

from ml.pipeline import read_raw
from ml.schema import FEATURES

BODY = dict(
    fieldOfStudy="Medicine", currentOccupation="Business Analyst", gender="Male", educationLevel="High School",
    industryGrowthRate="High", familyInfluence="High", age=48, yearsOfExperience=7, jobSatisfaction=7,
    workLifeBalance=10, jobOpportunities=83, salary=198266, jobSecurity=8, careerChangeInterest=0, skillsGap=8,
    mentorshipAvailable=0, certifications=0, freelancingExperience=0, geographicMobility=1,
    professionalNetworks=2, careerChangeEvents=0, technologyAdoption=1,
)


@pytest.fixture(scope="module")
def client(trained, monkeypatch_module):
    out, _ = trained
    monkeypatch_module.setenv("MODEL_DIR", str(out))
    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_contract(client):
    r = client.post("/predict", json=BODY)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in (0, 1)
    assert data["probability"] + data["probability_0"] == pytest.approx(1.0)
    assert len(data["top_features"]) == 3 and data["unseen_categories"] == []


def test_api_matches_batch_scoring(client, trained, spark, sample_csv):
    """Skew regression: a row scored through the API equals the same row scored in batch by the saved pipeline."""
    out, _ = trained
    model = PipelineModel.load(str(out / "career_change_pipeline"))
    for row in read_raw(spark, str(sample_csv)).limit(15).collect():
        body = {f.api: row[f.model] for f in FEATURES}
        batch = model.transform(spark.createDataFrame([row])).select("probability").first()[0][1]
        assert client.post("/predict", json=body).json()["probability"] == pytest.approx(batch, abs=1e-9)


def test_categories_change_the_prediction(client):
    """The first API version refitted its indexers on one row, so every category became index 0."""
    probs = {
        occ: client.post("/predict", json={**BODY, "currentOccupation": occ}).json()["probability"]
        for occ in ("Doctor", "Artist", "Lawyer", "Teacher")
    }
    assert len({round(p, 9) for p in probs.values()}) > 1


def test_unseen_category_is_reported_not_fatal(client):
    r = client.post("/predict", json={**BODY, "fieldOfStudy": "Astrology"})
    assert r.status_code == 200 and r.json()["unseen_categories"] == ["fieldOfStudy=Astrology"]


@pytest.mark.parametrize("patch", [{"age": 5}, {"jobSatisfaction": 11}, {"careerChangeInterest": 2},
                                   {"gender": ""}, {"salary": -1}, {"unexpected": 1}])
def test_invalid_input_is_rejected(client, patch):
    assert client.post("/predict", json={**BODY, **patch}).status_code == 422


def test_missing_field_is_rejected(client):
    body = dict(BODY)
    body.pop("salary")
    assert client.post("/predict", json=body).status_code == 422
