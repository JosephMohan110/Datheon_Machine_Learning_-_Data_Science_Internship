"""Reproducible California housing regression benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
TARGET_COLUMN = "median_house_value"
CATEGORICAL_FEATURES = ("ocean_proximity",)
BASE_NUMERIC_FEATURES = (
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
)
ENGINEERED_FEATURES = (
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
    "income_x_latitude",
    "income_x_longitude",
)
NUMERIC_FEATURES = (*BASE_NUMERIC_FEATURES, *ENGINEERED_FEATURES)
REQUIRED_COLUMNS = (*BASE_NUMERIC_FEATURES, TARGET_COLUMN, *CATEGORICAL_FEATURES)
RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_ALPHAS = (0.001, 0.002, 0.005, 0.01, 0.02)

sns.set_theme(style="whitegrid", context="notebook")


def load_housing_data(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate California housing records from a local CSV."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Housing dataset was not found: {data_path}")

    frame = pd.read_csv(data_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Dataset contains no rows.")

    for column in (*BASE_NUMERIC_FEATURES, TARGET_COLUMN):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    target = frame[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("Target values must be finite and cannot be missing.")
    if (target < 0).any():
        raise ValueError("Median house values must be non-negative.")
    for column in BASE_NUMERIC_FEATURES:
        values = frame[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Feature {column!r} contains non-finite numeric values.")
    frame[CATEGORICAL_FEATURES[0]] = (
        frame[CATEGORICAL_FEATURES[0]].astype("string").str.strip()
    )
    frame.loc[frame[CATEGORICAL_FEATURES[0]] == "", CATEGORICAL_FEATURES[0]] = pd.NA
    return frame


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add stateless, interpretable ratios and income/location interactions."""
    missing_columns = [
        column for column in BASE_NUMERIC_FEATURES if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Cannot engineer features; missing columns: {missing_columns}")

    engineered = frame.copy()

    def safe_ratio(numerator: str, denominator: str) -> np.ndarray:
        numerator_values = engineered[numerator].to_numpy(dtype=float, na_value=np.nan)
        denominator_values = engineered[denominator].to_numpy(dtype=float, na_value=np.nan)
        result = np.full(len(engineered), np.nan, dtype=float)
        np.divide(
            numerator_values,
            denominator_values,
            out=result,
            where=np.isfinite(denominator_values) & (denominator_values != 0),
        )
        return result

    engineered["rooms_per_household"] = safe_ratio("total_rooms", "households")
    engineered["bedrooms_per_room"] = safe_ratio("total_bedrooms", "total_rooms")
    engineered["population_per_household"] = safe_ratio("population", "households")
    engineered["income_x_latitude"] = engineered["median_income"] * engineered["latitude"]
    engineered["income_x_longitude"] = engineered["median_income"] * engineered["longitude"]
    return engineered


def split_dataset(
    frame: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split raw features and target before fitting any learned preprocessing."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dataset must contain target column {TARGET_COLUMN!r}.")
    X = frame.drop(columns=[TARGET_COLUMN])
    y = frame[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Small sklearn-compatible wrapper around deterministic housing features."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HousingFeatureEngineer":
        del y
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Housing feature engineering requires a pandas DataFrame.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Housing feature engineering requires a pandas DataFrame.")
        return engineer_features(X)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        original = list(input_features) if input_features is not None else list(BASE_NUMERIC_FEATURES)
        return np.asarray([*original, *ENGINEERED_FEATURES], dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Create train-fitted imputation, scaling, and categorical encoding."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(regressor: Any) -> Pipeline:
    """Assemble stateless feature engineering with train-fitted preprocessing."""
    return Pipeline(
        [
            ("feature_engineering", HousingFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )


def _build_search(
    estimator: Any,
    alphas: Sequence[float],
    cv: int,
    random_state: int,
) -> GridSearchCV:
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Grid-search alphas must be finite positive numbers.")
    pipeline = build_pipeline(estimator)
    return GridSearchCV(
        estimator=pipeline,
        param_grid={"regressor__regressor__alpha": list(alphas)},
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=cv, shuffle=True, random_state=random_state),
        n_jobs=1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )


def fit_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    cv: int = 5,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Fit OLS and cross-validated Ridge/Lasso; return held-out metrics."""
    if cv < 2:
        raise ValueError("Cross-validation requires at least two folds.")
    if len(X_train) < cv:
        raise ValueError(f"Training set needs at least {cv} rows for cross-validation.")

    model_specs: dict[str, tuple[Any, bool]] = {
        "OLS": (LinearRegression(), False),
        "Ridge": (
            TransformedTargetRegressor(
                regressor=Ridge(),
                transformer=StandardScaler(),
            ),
            True,
        ),
        "Lasso": (
            TransformedTargetRegressor(
                regressor=Lasso(max_iter=20_000, tol=1e-4),
                transformer=StandardScaler(),
            ),
            True,
        ),
    }
    results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, float | int | str | None]] = []
    cv_rows: list[dict[str, float | int | str]] = []

    for model_name, (regressor, tune_alpha) in model_specs.items():
        if tune_alpha:
            search = _build_search(regressor, alphas, cv, random_state)
            search.fit(X_train, y_train)
            estimator = search.best_estimator_
            best_alpha: float | None = float(
                search.best_params_["regressor__regressor__alpha"]
            )
            cv_rmse = float(-search.best_score_)
            for alpha, mean_score, std_score, rank in zip(
                search.cv_results_["param_regressor__regressor__alpha"],
                search.cv_results_["mean_test_score"],
                search.cv_results_["std_test_score"],
                search.cv_results_["rank_test_score"],
            ):
                cv_rows.append(
                    {
                        "model": model_name,
                        "alpha": float(alpha),
                        "cv_rmse_mean": float(-mean_score),
                        "cv_rmse_std": float(std_score),
                        "rank": int(rank),
                    }
                )
        else:
            estimator = build_pipeline(regressor)
            estimator.fit(X_train, y_train)
            best_alpha = None
            cv_rmse = float("nan")

        predictions = estimator.predict(X_test)
        residuals = np.asarray(y_test, dtype=float) - predictions
        fitted_regressor = getattr(
            estimator.named_steps["regressor"], "regressor_", estimator.named_steps["regressor"]
        )
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
            "cv_rmse": cv_rmse,
            "best_alpha": best_alpha,
            "nonzero_coefficients": int(
                np.count_nonzero(np.abs(fitted_regressor.coef_) > 1e-10)
            ),
        }
        results[model_name] = {
            "estimator": estimator,
            "predictions": predictions,
            "residuals": residuals,
            "metrics": metrics,
        }
        summary_rows.append({"model": model_name, **metrics})

    summary = pd.DataFrame(summary_rows).sort_values(
        "rmse", ascending=True, ignore_index=True
    )
    cv_results = pd.DataFrame(cv_rows).sort_values(
        ["model", "rank", "alpha"], ignore_index=True
    )
    return results, summary, cv_results


def plot_target_distribution(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot target distribution and expose the capped upper target value."""
    output_dir = Path(output_dir)
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sns.histplot(frame[TARGET_COLUMN], bins=60, kde=True, ax=axis, color="#286A8A")
    axis.axvline(
        frame[TARGET_COLUMN].max(),
        color="#B44A3A",
        linestyle="--",
        label=f"Observed maximum: ${frame[TARGET_COLUMN].max():,.0f}",
    )
    axis.set(title="Median house-value distribution", xlabel="Median house value (USD)")
    axis.legend()
    return _save_figure(fig, output_dir / "target_distribution.png")


def plot_correlation_heatmap(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot correlations among original/engineered numeric features and target."""
    engineered = engineer_features(frame)
    columns = [*NUMERIC_FEATURES, TARGET_COLUMN]
    correlation = engineered[columns].corr()
    fig, axis = plt.subplots(figsize=(13, 10), constrained_layout=True)
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        ax=axis,
    )
    axis.set_title("Numeric feature and target correlations", weight="bold")
    return _save_figure(fig, Path(output_dir) / "correlation_heatmap.png")


def plot_predictions(
    y_test: pd.Series,
    results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Compare holdout predictions to actual values for all three models."""
    count = len(results)
    fig, axes = plt.subplots(1, count, figsize=(6 * count, 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    actual = np.asarray(y_test, dtype=float)
    lower = min(float(actual.min()), *(float(r["predictions"].min()) for r in results.values()))
    upper = max(float(actual.max()), *(float(r["predictions"].max()) for r in results.values()))

    for axis, (model_name, result) in zip(axes, results.items()):
        sns.scatterplot(
            x=actual,
            y=result["predictions"],
            alpha=0.28,
            s=18,
            edgecolor=None,
            ax=axis,
        )
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#B44A3A")
        axis.set(
            title=model_name,
            xlabel="Actual value (USD)",
            ylabel="Predicted value (USD)",
            xlim=(lower, upper),
            ylim=(lower, upper),
        )
        axis.ticklabel_format(style="plain")
    fig.suptitle("Holdout predictions vs actual values", weight="bold")
    return _save_figure(fig, Path(output_dir) / "predicted_vs_actual.png")


def plot_residuals(
    results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot residual distributions and residuals against fitted values."""
    count = len(results)
    fig, axes = plt.subplots(2, count, figsize=(6 * count, 9), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, count)
    for column, (model_name, result) in enumerate(results.items()):
        residuals = result["residuals"]
        predictions = result["predictions"]
        sns.histplot(
            residuals,
            bins=50,
            kde=True,
            color="#286A8A",
            ax=axes[0, column],
        )
        axes[0, column].axvline(0, color="#B44A3A", linestyle="--")
        axes[0, column].set(
            title=f"{model_name}: residual distribution",
            xlabel="Residual (actual - predicted, USD)",
        )
        sns.scatterplot(
            x=predictions,
            y=residuals,
            alpha=0.25,
            s=16,
            edgecolor=None,
            ax=axes[1, column],
        )
        axes[1, column].axhline(0, color="#B44A3A", linestyle="--")
        axes[1, column].set(
            title=f"{model_name}: residuals vs fitted",
            xlabel="Predicted value (USD)",
            ylabel="Residual (USD)",
        )
        axes[1, column].ticklabel_format(style="plain", axis="both")
    fig.suptitle("Holdout residual analysis", weight="bold")
    return _save_figure(fig, Path(output_dir) / "residual_analysis.png")


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def run_benchmark(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    cv: int = 5,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> dict[str, Any]:
    """Run the full regression benchmark and write charts, metrics, and metadata."""
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    frame = load_housing_data(data_path)
    X_train, X_test, y_train, y_test = split_dataset(frame)
    results, summary, cv_results = fit_and_evaluate(
        X_train, X_test, y_train, y_test, cv=cv, alphas=alphas
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_dir / "model_metrics.csv", index=False)
    cv_results.to_csv(output_dir / "regularization_cv_results.csv", index=False)
    residual_summary = {
        model_name: {
            "mean_residual": float(np.mean(result["residuals"])),
            "median_residual": float(np.median(result["residuals"])),
            "residual_std": float(np.std(result["residuals"], ddof=1)),
            "within_50000_usd": float(
                np.mean(np.abs(result["residuals"]) <= 50_000)
            ),
        }
        for model_name, result in results.items()
    }
    (output_dir / "residual_summary.json").write_text(
        json.dumps(residual_summary, indent=2), encoding="utf-8"
    )

    target_cap = float(frame[TARGET_COLUMN].max())
    metadata = {
        "dataset": data_path.name,
        "rows": int(len(frame)),
        "feature_columns": list(frame.drop(columns=[TARGET_COLUMN]).columns),
        "target": TARGET_COLUMN,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cross_validation_folds": cv,
        "alpha_grid": [float(alpha) for alpha in alphas],
        "missing_values_by_column": {
            column: int(count) for column, count in frame.isna().sum().items()
        },
        "target_observed_max": target_cap,
        "rows_at_observed_target_max": int((frame[TARGET_COLUMN] == target_cap).sum()),
        "engineered_features": list(ENGINEERED_FEATURES),
        "preprocessing": {
            "numeric_imputation": "training-fold median",
            "numeric_scaling": "StandardScaler",
            "categorical_imputation": "training-fold most frequent",
            "categorical_encoding": "OneHotEncoder(drop='first', handle_unknown='ignore')",
        },
    }
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    artifacts = {
        "target_distribution": plot_target_distribution(frame, output_dir),
        "correlation_heatmap": plot_correlation_heatmap(frame, output_dir),
        "predicted_vs_actual": plot_predictions(y_test, results, output_dir),
        "residual_analysis": plot_residuals(results, output_dir),
    }
    return {
        "data": frame,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "results": results,
        "summary": summary,
        "cv_results": cv_results,
        "residual_summary": residual_summary,
        "metadata": metadata,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark OLS, Ridge, and Lasso on California housing data."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to housing CSV (default: data.csv beside this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics and evaluation charts.",
    )
    parser.add_argument("--cv", type=int, default=5, help="Training-fold CV splits.")
    args = parser.parse_args()

    run = run_benchmark(args.data, args.output_dir, cv=args.cv)
    print("Holdout metrics (lower RMSE/MAE and higher R2 are better):")
    print(run["summary"].to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(f"\nEvaluation artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
