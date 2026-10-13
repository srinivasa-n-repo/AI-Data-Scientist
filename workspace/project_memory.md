Target column: Price
Business goal: Predict fair car price
Metric: RMSE, MAE, R2 (in original Price units)
Exclusions: None
Data file: ToyotaCorolla.csv
Cleaning decisions: Dropped duplicates, IQR outlier removal on Age, KM, HP, CC, Doors, Weight
Features chosen: Price (transformed), Age, KM, HP, MetColor, Automatic, CC, Doors, Weight, FuelType_Diesel, FuelType_Petrol
Models chosen: LinearRegression, RandomForest, XGBoost
Latest metrics: 
- LinearRegression: RMSE=0.55, MAE=0.39, R2=0.72
- RandomForest: RMSE=0.56, MAE=0.37, R2=0.71
- XGBoost: RMSE=0.57, MAE=0.40, R2=0.70