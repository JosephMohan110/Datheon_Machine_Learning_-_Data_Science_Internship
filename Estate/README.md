# California Housing Price Regression

A reproducible regression project comparing Ordinary Least Squares (OLS), Ridge (L2), and Lasso (L1) on the local California housing CSV. It includes feature engineering, leakage-safe imputation and scaling, categorical encoding, cross-validated regularization selection, holdout metrics, and residual diagnostics.

## Project layout

```text
Estate/
├── data.csv
├── housing_regression.py
├── notebooks/
│   └── housing_price_regression.ipynb
├── outputs/
│   ├── benchmark_metadata.json
│   ├── correlation_heatmap.png
│   ├── model_metrics.csv
│   ├── predicted_vs_actual.png
│   ├── regularization_cv_results.csv
│   ├── residual_analysis.png
│   ├── residual_summary.json
│   └── target_distribution.png
├── requirements.txt
└── tests/
    └── test_housing_regression.py
```

## Setup on Windows

From PowerShell in this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

To reuse the existing parent-folder virtual environment instead:

```powershell
..\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the notebook

From the `Estate` folder:

```powershell
jupyter lab
```

Open `notebooks/housing_price_regression.ipynb` and run all cells. The executed notebook includes dataset inspection, exploratory plots, the data split, cross-validated model selection, holdout metrics, and residual analysis.

## Run the benchmark script

From the `Estate` folder:

```powershell
python housing_regression.py
```

The default run uses five-fold training-only cross-validation over five alpha values (`0.001` through `0.02`) for each regularized model. It writes CSV/JSON summaries and PNG plots to `outputs/`. You can override the input data and output folder:

```powershell
python housing_regression.py --data data.csv --output-dir outputs --cv 5
```

No hosted API or network request is used by the benchmark.

## Methodology

- The local CSV contains 20,640 records, eight numeric inputs, the `ocean_proximity` categorical input, and `median_house_value` as the target.
- The target has no missing values. `total_bedrooms` has 207 missing values; numeric values are median-imputed using training-fold statistics, while categorical values are imputed using the most frequent training-fold category.
- Feature engineering adds rooms per household, bedrooms per room, population per household, and income-by-latitude/longitude interaction terms. These transformations are deterministic and fitted preprocessing (imputation/scaling/encoding) remains inside the estimator pipeline.
- Numeric inputs are standardized. `ocean_proximity` is one-hot encoded with one category dropped and unseen categories ignored.
- The data is split once into 80% training and 20% holdout using seed `42`. Five-fold `GridSearchCV` selects alpha on training data only using negative RMSE. The holdout is not used to select hyperparameters.
- Ridge and Lasso standardize the target inside `TransformedTargetRegressor` during fitting, then inverse-transform predictions to USD. This makes alpha search numerically better scaled while keeping reported metrics in the original target units.
- All models report holdout RMSE, MAE, and R². Lower RMSE/MAE and higher R² are better. The residual plots use residual = actual minus predicted.
- Regularization mitigates coefficient instability from correlated features. The correlation heatmap includes original and engineered numeric inputs; it is descriptive, not a feature-selection procedure.

## Important dataset limitation

The observed target maximum is $500,001, and many records equal that maximum. This dataset caps high home values, so the true values above the cap are not observed. Predictions at the expensive end are therefore trained on censored labels; metrics and residuals should be interpreted with this limitation in mind.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests cover the observed dataset characteristics, feature engineering, missing-value handling through the modeling pipeline, cross-validation results, and output artifact creation.

## Holdout results

Results from the complete five-fold alpha search with `random_state=42`:

| Model | Test RMSE (USD) | Test MAE (USD) | Test R² | Best alpha | CV RMSE (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| OLS | 68,679.61 | 49,133.69 | 0.640 | — | — |
| Ridge | 68,687.33 | 49,144.61 | 0.640 | 0.010 | 67,584.68 |
| Lasso | 69,151.10 | 49,621.15 | 0.635 | 0.001 | 67,865.09 |

These are from one random holdout; the target cap and the lack of geographic/time-based validation limit what can be concluded from this comparison.

## Interpretation

This is a benchmark on one historical dataset, not a production valuation system. A random split does not measure geographic or future-time generalization. For deployment, use appropriate time/geographic validation, assess drift and target censoring, and investigate large errors before relying on estimates.
