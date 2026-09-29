import csv

from api.main import PredictionRequest
from ml.schema import BY_API, CATEGORICAL, FEATURES, LABEL_CSV, NUMERIC

from .conftest import RAW_CSV


def test_schema_matches_csv_header():
    header = next(csv.reader(RAW_CSV.open()))
    assert sorted(header) == sorted([f.csv for f in FEATURES] + [LABEL_CSV])


def test_names_are_unique_and_snake_case():
    assert len({f.model for f in FEATURES}) == len(FEATURES)
    assert all(f.model == f.model.lower() and " " not in f.model and "-" not in f.model for f in FEATURES)


def test_api_contract_equals_schema():
    """The pydantic request model and the training schema must list exactly the same fields."""
    assert set(PredictionRequest.model_fields) == set(BY_API)


def test_kinds_partition_features():
    assert len(CATEGORICAL) + len(NUMERIC) == len(FEATURES)
