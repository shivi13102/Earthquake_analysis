import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Load Dataset
data = pd.read_csv('Final_clean_data.csv')

# Select features and target variables
features = ['magnitude', 'depth', 'latitude', 'longitude', 'refined_alert', 'tsunami', 'sig', 'dmin', 'gap']
X = data[features]
y = data[['cdi', 'mmi']]

# Convert refined_alert to numerical values
alert_mapping = {'green': 0, 'yellow': 1, 'orange': 2, 'red': 3}
X['refined_alert'] = X['refined_alert'].map(alert_mapping).fillna(0)  # Fill missing alerts with 0

# Handling missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Use QuantileTransformer for CDI
transformer = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
y['cdi'] = transformer.fit_transform(y[['cdi']])

# Normalize/Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": MultiOutputRegressor(RandomForestRegressor(random_state=42)),
    "XGBoost": MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', random_state=42)),
    "LightGBM": MultiOutputRegressor(lgb.LGBMRegressor(random_state=42)),
    "CatBoost": MultiOutputRegressor(CatBoostRegressor(random_state=42, verbose=0)),
    "MultiOutput Ridge": MultiOutputRegressor(Ridge(random_state=42)),
    "MultiOutput SVR": MultiOutputRegressor(SVR()),
    "Neural Network": MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500, early_stopping=True))
}

best_model = None
best_score = float('inf')
best_model_name = ""

# Train and evaluate models
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Inverse transform CDI predictions
    y_pred[:, 0] = transformer.inverse_transform(y_pred[:, [0]]).flatten()
    y_test['cdi'] = transformer.inverse_transform(y_test[['cdi']]).flatten()
    
    # Ensure CDI values remain non-negative
    y_test['cdi'] = y_test['cdi'].clip(0)
    y_pred[:, 0] = y_pred[:, 0].clip(0)

    # Compute MSE
    cdi_mse = mean_squared_error(y_test['cdi'], y_pred[:, 0])
    mmi_mse = mean_squared_error(y_test['mmi'], y_pred[:, 1])
    total_mse = cdi_mse + mmi_mse

    # Compute R² score
    cdi_r2 = r2_score(y_test['cdi'], y_pred[:, 0])
    mmi_r2 = r2_score(y_test['mmi'], y_pred[:, 1])
    avg_r2 = (cdi_r2 + mmi_r2) / 2  # Average R² score

    print(f"{name} - CDI MSE: {cdi_mse:.4f}, MMI MSE: {mmi_mse:.4f}, Total MSE: {total_mse:.4f}")
    print(f"{name} - CDI R²: {cdi_r2:.4f}, MMI R²: {mmi_r2:.4f}, Average R²: {avg_r2:.4f}")

    # Track the best model based on MSE
    if total_mse < best_score:
        best_score = total_mse
        best_model = model
        best_model_name = name

# Print best model
print(f"\nBest Model: {best_model_name} with Total MSE: {best_score:.4f}")

# Save the best model and transformers
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(transformer, 'transformer.pkl')

print("Best model saved successfully.")