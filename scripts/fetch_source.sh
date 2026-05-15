#!/usr/bin/env bash
set -euo pipefail

BUCKET="s3://thesis-data-ismaktam"
ROOT="$(cd "$(dirname "$0")/.." && pwd)/source"
mkdir -p "$ROOT"/{kriging/kfold,kriging/test,lgbm,bayesnf/final,dataset}

echo "==> Kriging"
aws s3 cp "$BUCKET/kriging/global_variogram.pkl"             "$ROOT/kriging/global_variogram.pkl"
aws s3 cp "$BUCKET/kriging/winner.json"                      "$ROOT/kriging/winner.json"
aws s3 cp "$BUCKET/kriging/holdout_station_ids.json"         "$ROOT/kriging/holdout_station_ids.json"
aws s3 cp "$BUCKET/kriging/kfold/cv_results.json"            "$ROOT/kriging/kfold/cv_results.json"
aws s3 cp "$BUCKET/kriging/test/holdout_predictions.parquet" "$ROOT/kriging/test/holdout_predictions.parquet"
aws s3 cp "$BUCKET/kriging/test/test_metrics.json"           "$ROOT/kriging/test/test_metrics.json"

echo "==> LGBM"
aws s3 cp --recursive \
    --exclude "*" \
    --include "final/stage1.joblib" \
    --include "final/models.joblib" \
    --include "final/features.parquet" \
    --include "hparam/*" \
    --exclude "hparam/hpo_features.parquet" \
    --include "kfold/cv_results.json" \
    --include "fold_assignment.parquet" \
    --include "oof_predictions.parquet" \
    --include "uncertainty_fold0.json" \
    "$BUCKET/lgbm/" "$ROOT/lgbm/"

echo "==> BayesNF"
aws s3 cp --recursive \
    "$BUCKET/bayesnf/runs/vi__final__WY_h1_10__ffrk_full/" \
    "$ROOT/bayesnf/runs/vi__final__WY_h1_10__ffrk_full/"
aws s3 cp "$BUCKET/bayesnf/final/features.parquet"   "$ROOT/bayesnf/final/features.parquet"
aws s3 cp "$BUCKET/bayesnf/kfold_summary_vi.json"    "$ROOT/bayesnf/kfold_summary_vi.json"
aws s3 cp "$BUCKET/bayesnf/kfold_summary_map.json"   "$ROOT/bayesnf/kfold_summary_map.json"

echo "==> Multi-resolution gridded product (worked-example month, March 2023)"
# Static features at coarser resolutions; at 1 km the static columns are embedded in each monthly file.
for RES in 2 5 10; do
  aws s3 cp "$BUCKET/dataset/grid_${RES}km_static.parquet" \
            "$ROOT/dataset/grid_${RES}km_static.parquet"
done
# Only March 2023 — the worked-example month referenced from Chapter 7 / Chapter 8.
for RES in 1 2 5 10; do
  aws s3 cp "$BUCKET/dataset/grid_${RES}km/2023-03.parquet" \
            "$ROOT/dataset/grid_${RES}km/2023-03.parquet"
  aws s3 cp "$BUCKET/dataset/predictions_${RES}km/2023-03.parquet" \
            "$ROOT/dataset/predictions_${RES}km/2023-03.parquet"
done

echo "==> Sanity report"
du -sh "$ROOT"/*/ | sort -h
echo "Total files: $(find "$ROOT" -type f | wc -l)"
echo "Total size:  $(du -sh "$ROOT" | cut -f1)"
