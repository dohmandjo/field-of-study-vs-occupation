"""Feature pipeline + model definition.

The whole preprocessing chain (StringIndexer -> OneHotEncoder -> VectorAssembler) is part of ONE
``PipelineModel`` that is fitted on the training split and saved with the classifier. The API loads that
exact object, so a request is encoded with the categories learned during training. (The first version of
the API refitted its indexers on the single request row, which mapped every category to index 0.)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.stat import Summarizer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .schema import BY_MODEL, CATEGORICAL, FEATURES, LABEL, LABEL_CSV, NUMERIC

FEATURES_COL = "features"


def get_spark(app_name: str = "career-change", master: str = "local[2]") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def read_raw(spark: SparkSession, path: str) -> DataFrame:
    """Read the raw CSV and rename its headers to the snake_case names used by the pipeline."""
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    renames = {f.csv: f.model for f in FEATURES}
    renames[LABEL_CSV] = LABEL
    missing = set(renames) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
    for old, new in renames.items():
        df = df.withColumnRenamed(old, new)
    return df.select([*renames.values()]).dropna()


def feature_columns(exclude: Iterable[str] = ()) -> tuple[list[str], list[str]]:
    """(categorical, numeric) model column names after removing ``exclude`` (API names)."""
    drop = set(exclude)
    unknown = drop - {f.api for f in FEATURES}
    if unknown:
        raise ValueError(f"Unknown feature(s) to exclude: {sorted(unknown)}")
    cats = [f.model for f in CATEGORICAL if f.api not in drop]
    nums = [f.model for f in NUMERIC if f.api not in drop]
    return cats, nums


def build_pipeline(exclude: Sequence[str] = (), max_iter: int = 100) -> Pipeline:
    cats, nums = feature_columns(exclude)
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cats]
    # handleInvalid="keep" gives unseen categories their own ``__unknown`` slot. Its coefficient stays ~0
    # because training never sees it, so an unseen category no longer crashes a request; it just adds no signal.
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cats],
        outputCols=[f"{c}_ohe" for c in cats],
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=[f"{c}_ohe" for c in cats] + nums, outputCol=FEATURES_COL, handleInvalid="error"
    )
    classifier = LogisticRegression(featuresCol=FEATURES_COL, labelCol=LABEL, maxIter=max_iter)
    return Pipeline(stages=[*indexers, encoder, assembler, classifier])


def split(df: DataFrame, seed: int = 42, test_fraction: float = 0.2) -> tuple[DataFrame, DataFrame]:
    train, test = df.randomSplit([1 - test_fraction, test_fraction], seed=seed)
    return train, test


def evaluate(model: PipelineModel, test: DataFrame) -> dict:
    """Metrics on a held-out split.

    AUC must be computed from ``rawPrediction`` (a continuous score). The original notebook passed the hard
    0/1 ``prediction`` column, which collapses the ROC curve to a single point and under-reports AUC.
    """
    scored = model.transform(test).cache()
    auc = BinaryClassificationEvaluator(
        labelCol=LABEL, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(scored)
    acc = MulticlassClassificationEvaluator(
        labelCol=LABEL, predictionCol="prediction", metricName="accuracy"
    ).evaluate(scored)
    f1 = MulticlassClassificationEvaluator(
        labelCol=LABEL, predictionCol="prediction", metricName="f1"
    ).evaluate(scored)
    n = scored.count()
    positives = scored.where(F.col(LABEL) == 1).count()
    confusion = {
        f"actual_{a}_pred_{p}": scored.where((F.col(LABEL) == a) & (F.col("prediction") == p)).count()
        for a in (0, 1)
        for p in (0, 1)
    }
    scored.unpersist()
    return {
        "test_rows": n,
        "auc_roc": round(auc, 4),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "majority_class_baseline": round(max(positives, n - positives) / n, 4),
        "confusion_matrix": confusion,
    }


def slot_names(assembled: DataFrame) -> list[str]:
    """Name of each slot of the ``features`` vector, read from the column metadata of a transformed frame."""
    meta = assembled.schema[FEATURES_COL].metadata["ml_attr"]["attrs"]
    slots = {a["idx"]: a["name"] for kind in meta.values() for a in kind}
    return [slots[i] for i in range(len(slots))]


def feature_names(model: PipelineModel, df: DataFrame) -> list[str]:
    return slot_names(model.transform(df.limit(1)))


def feature_means(model: PipelineModel, df: DataFrame) -> list[float]:
    """Mean of every feature slot on ``df`` (used to centre contributions when explaining a prediction)."""
    row = model.transform(df).select(Summarizer.mean(F.col(FEATURES_COL)).alias("m")).first()
    return [float(x) for x in row["m"].toArray()]


def explain(names: Sequence[str], coefs: Sequence[float], means: Sequence[float],
            values: Sequence[float], top: int = 3) -> list[dict]:
    """Per-feature push on the log-odds of "likely to change", relative to an average training row.

    contribution_slot = coef * (x - mean); one-hot slots are summed back into their original feature.
    Positive means the value makes a change more likely than for the average person.
    """
    per_feature: dict[str, float] = defaultdict(float)
    for name, c, m, x in zip(names, coefs, means, values, strict=True):
        column = name.split("_ohe_")[0]
        per_feature[column] += c * (x - m)
    ranked = sorted(per_feature.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top]
    return [{"feature": BY_MODEL[col].csv, "effect": round(v, 4)} for col, v in ranked]


def coefficients_array(model: PipelineModel) -> list[float]:
    return [float(c) for c in model.stages[-1].coefficients.toArray()]


def coefficients(model: PipelineModel, df: DataFrame) -> list[dict]:
    """Coefficients sorted by magnitude (features are on different scales, so read them as direction)."""
    names = feature_names(model, df)
    coefs = model.stages[-1].coefficients.toArray()
    if len(coefs) != len(names):
        raise RuntimeError("feature names and coefficients are out of sync")
    rows = [{"feature": n, "coefficient": float(c)} for n, c in zip(names, coefs, strict=True)]
    return sorted(rows, key=lambda r: abs(r["coefficient"]), reverse=True)
