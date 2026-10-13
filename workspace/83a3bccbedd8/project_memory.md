Project goal: predict fair price (regression). Target column: Price. Evaluation metric: RMSE (in original currency units). Data source: ToyotaCorolla.csv (1435 rows after cleaning). Feature engineering: label-encoded Price, one-hot encoded FuelType, Yeo-Johnson transform on numeric features (Age, KM, HP, MetColor, Automatic, CC, Doors, Weight). Train-val random split 80/20. Models trained: LinearRegression, RandomForest, XGBoost. Validation metrics (original Price units):
- LinearRegression: RMSE=1382, MAE=959, R2=0.841
- RandomForest: RMSE=1204, MAE=869, R2=0.879
- XGBoost: RMSE=1223, MAE=906, R2=0.875
Chosen model: RandomForest (lowest RMSE). Error size: average prediction error ~1200 currency units (~12% of mean price ~10000). Saved model_comparison.png.