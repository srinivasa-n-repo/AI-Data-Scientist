# Toyota Corolla Price Prediction Analysis Report

## Data Summary
- **Raw shape**: `(1436, 10)`
- **Columns**: `Price` (target), `Age`, `KM`, `FuelType`, `HP`, `MetColor`, `Automatic`, `CC`, `Doors`, `Weight`
- **Data types**: Nine numeric columns and one categorical (`FuelType`).

## Cleaning Decisions
- **Missing values**: None found in any column.
- **Duplicate rows**: One duplicate row identified and removed.
- **IQR outlier removal**: Applied to eligible numeric columns (`Age`, `KM`, `HP`, `CC`, `Doors`, `Weight`), removing **182** rows in total. Excluded binary columns (`MetColor`, `Automatic`) and target (`Price`) from outlier removal.
- **Final cleaned shape**: `(1253, 10)`.

## Features Used
- **Highly Skewed Features (Yeo-Johnson transformation)**: `Price`, `Automatic`, `Weight`
- **Moderately Skewed Features (Square Root transformation)**: `KM`, `HP`, `CC`
- **Negatively Skewed Features (Left as-is)**: `Age`, `MetColor`, `Doors`
- **Categorical Encoding**: `FuelType` one-hot encoded to `FuelType_Diesel` and `FuelType_Petrol`
- **Final Feature List**: `Price`, `Age`, `KM`, `HP`, `MetColor`, `Automatic`, `CC`, `Doors`, `Weight`, `FuelType_Diesel`, `FuelType_Petrol`

## Models Compared
- **LinearRegression**
- **RandomForest**
- **XGBoost**

## Metrics
| Model           | RMSE  | MAE  | R²   |
|-----------------|-------|------|------|
| LinearRegression| 0.55  | 0.39 | 0.72 |
| RandomForest    | 0.56  | 0.37 | 0.71 |
| XGBoost         | 0.57  | 0.40 | 0.70 |

## Plot References
`eda_distributions_and_outliers.png` — combined histograms and boxplots for numeric columns showing skewness and outliers.
`feature_distributions_transformed.png` — histograms of transformed features showing improved symmetry.
`model_comparison.png` — scatter plots of actual vs predicted prices for each model.

## Conclusion
LinearRegression performed best with RMSE = 0.55, MAE = 0.39, and R² = 0.72. The model explains most variance and is simple and interpretable. Further improvements could involve interaction terms or hyperparameter tuning.
