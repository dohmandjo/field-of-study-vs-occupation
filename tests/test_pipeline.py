import pytest
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F

from ml.pipeline import build_pipeline, coefficients, explain, feature_columns, feature_names, read_raw, split
from ml.schema import LABEL, LEAKY_FEATURE


def test_read_raw_renames_columns(spark, sample_csv):
    df = read_raw(spark, str(sample_csv))
    assert LABEL in df.columns and "work_life_balance" in df.columns
    assert df.count() == 3000


def test_exclude_drops_feature_and_rejects_unknown():
    cats, nums = feature_columns([LEAKY_FEATURE])
    assert "career_change_interest" not in nums and len(nums) == 15
    with pytest.raises(ValueError):
        feature_columns(["notAFeature"])


def test_model_beats_majority_baseline(trained):
    _, metrics = trained
    assert metrics["accuracy"] > metrics["majority_class_baseline"] + 0.1
    assert 0.5 < metrics["auc_roc"] <= 1.0


def test_auc_uses_scores_not_hard_labels(spark, sample_csv):
    """Regression for the notebook bug: AUC from the 0/1 `prediction` column under-reports the model."""
    df = read_raw(spark, str(sample_csv))
    train_df, test_df = split(df)
    model = build_pipeline(exclude=[LEAKY_FEATURE]).fit(train_df)
    scored = model.transform(test_df)
    ev = dict(labelCol=LABEL, metricName="areaUnderROC")
    proper = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", **ev).evaluate(scored)
    hard = BinaryClassificationEvaluator(rawPredictionCol="prediction", **ev).evaluate(scored)
    assert proper > hard + 0.02


def test_saved_pipeline_scores_unseen_category(trained, spark, sample_csv):
    out, _ = trained
    model = PipelineModel.load(str(out / "career_change_pipeline"))
    df = read_raw(spark, str(sample_csv)).limit(5).withColumn("gender", F.lit("Robot"))
    probs = model.transform(df).select("probability").collect()
    assert len(probs) == 5 and all(0 <= p[0][1] <= 1 for p in probs)


def test_feature_names_align_with_coefficients(trained, spark, sample_csv):
    out, _ = trained
    model = PipelineModel.load(str(out / "career_change_pipeline"))
    df = read_raw(spark, str(sample_csv))
    names = feature_names(model, df)
    assert len(names) == model.stages[-1].numFeatures
    assert "gender_ohe_Female" in names and "career_change_interest" in names
    assert coefficients(model, df)[0]["feature"] == "career_change_interest"   # the leaky feature dominates


def test_explain_groups_one_hot_slots_and_centres_values():
    names = ["gender_ohe_Female", "gender_ohe_Male", "salary"]
    effects = explain(names, coefs=[1.0, -1.0, 0.001], means=[0.5, 0.5, 100.0], values=[1.0, 0.0, 100.0])
    assert effects[0] == {"feature": "Gender", "effect": 1.0}          # (1*(1-.5)) + (-1*(0-.5))
    assert {"feature": "Salary", "effect": 0.0} in effects             # x == mean -> no push
