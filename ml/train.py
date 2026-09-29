"""Train the career-change classifier and save it.

    python -m ml.train                                   # all features (default)
    python -m ml.train --exclude careerChangeInterest    # honest variant: drop the leaky feature

Outputs (default ``models/``):
    career_change_pipeline/   a Spark PipelineModel (encoders + assembler + logistic regression)
    metrics.json              hold-out metrics, coefficients and the training options
    feature_means.json        training mean of each feature slot (the API uses it to explain a prediction)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .pipeline import build_pipeline, coefficients, evaluate, feature_means, get_spark, read_raw, split
from .schema import LEAKY_FEATURE

ROOT = Path(__file__).resolve().parent.parent


def train(data: str, out_dir: str, exclude: list[str], seed: int = 42, max_iter: int = 100) -> dict:
    spark = get_spark("career-change-train")   # reuses a running session; the caller owns its lifecycle
    spark.sparkContext.setLogLevel("ERROR")
    df = read_raw(spark, data)
    train_df, test_df = split(df, seed=seed)

    model = build_pipeline(exclude=exclude, max_iter=max_iter).fit(train_df)
    metrics = evaluate(model, test_df)
    metrics.update(
        {
            "train_rows": train_df.count(),
            "seed": seed,
            "excluded_features": exclude,
            "coefficients_top10": coefficients(model, test_df)[:10],
        }
    )

    out = Path(out_dir)
    model_path = out / "career_change_pipeline"
    if model_path.exists():
        shutil.rmtree(model_path)
    out.mkdir(parents=True, exist_ok=True)
    model.write().save(str(model_path))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "feature_means.json").write_text(json.dumps(feature_means(model, train_df)))
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data/raw/field_of_study_vs_occupation.csv"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--exclude", nargs="*", default=[], metavar="FEATURE",
                    help=f"API feature names to drop, e.g. {LEAKY_FEATURE}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=100)
    args = ap.parse_args(argv)
    try:
        metrics = train(args.data, args.out, args.exclude, args.seed, args.max_iter)
    finally:
        get_spark().stop()
    print(json.dumps({k: v for k, v in metrics.items() if k != "coefficients_top10"}, indent=2))


if __name__ == "__main__":
    main()
