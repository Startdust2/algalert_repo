import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import time

# Load the data
df = pd.read_csv('records_subset_labeled (1).csv', nrows=2074)

# Features and target for regression (excluding 'Chl_a', 'bloomstatus', and the final column)
X_reg = df.drop(['Chl_a', 'bloomstatus'], axis=1).iloc[:, 1:].values
y_reg = df['Chl_a'].values

# Data scaling for regression
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

# Reshape y_reg to a 2D array before scaling
y_reg = y_reg.reshape(-1, 1)  # Reshape to a 2D array
scaler_y_reg = StandardScaler()
y_reg_scaled = scaler_y_reg.fit_transform(y_reg)

# Define the regression models to test
models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='rbf'),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0)
}

# Perform cross-validation for each model and print the results
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    start_time = time.time()
    
    cv_mse_scores = cross_val_score(model, X_reg_scaled, y_reg_scaled.ravel(), cv=kf, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(-cv_mse_scores)
    cv_mae_scores = cross_val_score(model, X_reg_scaled, y_reg_scaled.ravel(), cv=kf, scoring='neg_mean_absolute_error')
    cv_r2_scores = cross_val_score(model, X_reg_scaled, y_reg_scaled.ravel(), cv=kf, scoring='r2')
    
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n{model_name} Cross-Validation Metrics:")
    print(f"Mean Squared Error (MSE): {np.mean(-cv_mse_scores)}")
    print(f"Root Mean Squared Error (RMSE): {np.mean(cv_rmse_scores)}")
    print(f"Mean Absolute Error (MAE): {np.mean(-cv_mae_scores)}")
    print(f"R² Score: {np.mean(cv_r2_scores)}")
    print(f"Execution Time: {execution_time} seconds")
