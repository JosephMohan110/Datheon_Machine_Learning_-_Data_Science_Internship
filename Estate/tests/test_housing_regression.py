from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from housing_regression import (
    DEFAULT_ALPHAS,
    ENGINEERED_FEATURES,
    TARGET_COLUMN,
    build_pipeline,
    engineer_features,
    fit_and_evaluate,
    load_housing_data,
    run_benchmark,
    split_dataset,
)
from sklearn.linear_model import Ridge


PROJECT_DIR = Path(__file__).resolve().parents[1]


class HousingRegressionTests(unittest.TestCase):
    def test_dataset_schema_and_missing_bedrooms_are_preserved(self) -> None:
        frame = load_housing_data(PROJECT_DIR / "data.csv")

        self.assertEqual(len(frame), 20_640)
        self.assertEqual(frame["total_bedrooms"].isna().sum(), 207)
        self.assertEqual(frame["ocean_proximity"].nunique(), 5)
        self.assertFalse(frame[TARGET_COLUMN].isna().any())
        self.assertEqual(int((frame[TARGET_COLUMN] == 500_001).sum()), 965)

    def test_engineered_features_handle_missing_values_without_mutating_input(self) -> None:
        data = load_housing_data(PROJECT_DIR / "data.csv").iloc[:30].copy()
        data.loc[data.index[0], "total_bedrooms"] = np.nan
        original_columns = list(data.columns)
        engineered = engineer_features(data)

        self.assertEqual(list(data.columns), original_columns)
        self.assertTrue(set(ENGINEERED_FEATURES).issubset(engineered.columns))
        self.assertEqual(engineered["bedrooms_per_room"].isna().sum(), 1)
        self.assertTrue(np.isfinite(engineered["rooms_per_household"]).all())

    def test_pipeline_imputes_missing_values_and_predicts(self) -> None:
        frame = load_housing_data(PROJECT_DIR / "data.csv").iloc[:500].copy()
        frame.loc[frame.index[0], "total_bedrooms"] = np.nan
        X_train, X_test, y_train, _ = split_dataset(frame, random_state=7)
        model = build_pipeline(Ridge(alpha=1.0))
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        self.assertEqual(predictions.shape, (len(X_test),))
        self.assertTrue(np.isfinite(predictions).all())

    def test_model_selection_and_full_run_write_artifacts(self) -> None:
        frame = load_housing_data(PROJECT_DIR / "data.csv").sample(
            n=600, random_state=42
        ).reset_index(drop=True)
        X_train, X_test, y_train, y_test = split_dataset(frame)
        results, summary, cv_results = fit_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            cv=2,
            alphas=(0.1, 1.0),
        )

        self.assertEqual(set(results), {"OLS", "Ridge", "Lasso"})
        self.assertEqual(set(summary["model"]), {"OLS", "Ridge", "Lasso"})
        self.assertEqual(len(cv_results), 4)
        self.assertTrue(summary["rmse"].map(np.isfinite).all())
        self.assertTrue(summary["mae"].map(np.isfinite).all())
        self.assertTrue(summary["r2"].map(np.isfinite).all())

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            run = run_benchmark(
                PROJECT_DIR / "data.csv",
                output_dir,
                cv=2,
                alphas=(0.1,),
            )
            expected = {
                "benchmark_metadata.json",
                "model_metrics.csv",
                "regularization_cv_results.csv",
                "residual_summary.json",
                "target_distribution.png",
                "correlation_heatmap.png",
                "predicted_vs_actual.png",
                "residual_analysis.png",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected)
            self.assertEqual(len(run["summary"]), 3)
            self.assertTrue(all(path.is_file() for path in run["artifacts"].values()))


if __name__ == "__main__":
    unittest.main()
