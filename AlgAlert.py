#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:02:43 2024

@author: areejalsini
"""
#he code to save the DataFrame as a CSV file for each regression model after adding the `Chl_a_predicted` column. Here is the modified code:


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
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

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

# Save the scalers for later use
joblib.dump(scaler_reg, 'scaler_reg.pkl')
joblib.dump(scaler_y_reg, 'scaler_y_reg.pkl')

# Define the regression models to test
regression_models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='rbf'),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0)
}

# Perform regression and save the models and DataFrame with predictions
for model_name, reg_model in regression_models.items():
    # Train the regression model
    reg_model.fit(X_reg_scaled, y_reg_scaled.ravel())
    
    # Save the trained model to a file
    joblib.dump(reg_model, f'{model_name}.pkl')
    
    # Make predictions on the entire dataset
    reg_predictions = reg_model.predict(X_reg_scaled)
    
    # Inverse transform the predictions to original scale
    reg_predictions = scaler_y_reg.inverse_transform(reg_predictions.reshape(-1, 1)).ravel()
    
    # Add predicted 'Chl_a' values as a new feature for classification
    df['Chl_a_predicted'] = reg_predictions
    
    df_copy = df
    df_copy['bloomstatus']= df['bloomstatus']
    
    # Save the copied DataFrame with predictions to a CSV file
    df_copy.to_csv(f'df_with_{model_name}_predictions.csv', index=False)
    
    # Save the DataFrame with predictions to a CSV file
    #df.to_csv(f'df_with_{model_name}_predictions.csv', index=False)
    
    # Check if 'bloomstatus' is in the DataFrame before dropping it for classification
    if 'bloomstatus' in df.columns:
        # Preprocessing for classification
        X_cls = df.drop(['Chl_a', 'bloomstatus'], axis=1).iloc[:, 1:].values
        y_cls = df['bloomstatus'].values

        # Data scaling for classification
        scaler_cls = StandardScaler()
        X_cls_scaled = scaler_cls.fit_transform(X_cls)

        # Define a simple neural network model for classification
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_cls_scaled.shape[1],)),
            Dropout(0.5),  # Add dropout for regularization
            Dense(32, activation='relu'),
            Dropout(0.5),  # Add dropout for regularization
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Define early stopping based on validation accuracy
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True)

        # Perform k-fold cross-validation for classification (k=5)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        accuracy_scores = []
        
        for train_index, val_index in kf.split(X_cls_scaled):
            X_train_fold, X_val_fold = X_cls_scaled[train_index], X_cls_scaled[val_index]
            y_train_fold, y_val_fold = y_cls[train_index], y_cls[val_index]
            
            history = model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=100, callbacks=[early_stopping], verbose=0)
            
            val_accuracy = model.evaluate(X_val_fold, y_val_fold)[1]
            accuracy_scores.append(val_accuracy)
        
        print(f"\n{model_name} Regression Model - Classification Cross-Validation Accuracy: {np.mean(accuracy_scores)}")
        
        # Generate predictions on the entire dataset and print classification report using cross-validation predictions
        y_pred_probs = model.predict(X_cls_scaled)
        y_pred = (y_pred_probs > 0.5).astype(int)
        
        print("\nClassification Report:")
        print(classification_report(y_cls, y_pred))
    else:
        print("'bloomstatus' column not found in the DataFrame.")
