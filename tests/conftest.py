import csv
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_CSV = ROOT / "data" / "raw" / "field_of_study_vs_occupation.csv"


@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory) -> Path:
    """A reproducible 3,000-row sample so the whole suite trains in seconds."""
    with RAW_CSV.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        header = reader.fieldnames
    random.Random(0).shuffle(rows)
    path = tmp_path_factory.mktemp("data") / "sample.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows[:3000])
    return path


@pytest.fixture
def spark():
    """Function-scoped on purpose: the API's lifespan stops its Spark session on shutdown, and getOrCreate()
    transparently starts a new one, so no test depends on another test's session staying alive."""
    from ml.pipeline import get_spark

    session = get_spark("tests", master="local[2]")
    session.sparkContext.setLogLevel("ERROR")
    return session


@pytest.fixture(scope="session")
def trained(sample_csv, tmp_path_factory):
    """Train once through the real CLI code path; returns (model_dir, metrics)."""
    from ml.train import train

    out = tmp_path_factory.mktemp("models")
    metrics = train(str(sample_csv), str(out), exclude=[])
    return out, metrics
