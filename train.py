# train.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv('data/aqi_data_fetched1.csv')

# Combine date and time into datetime
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df = df.sort_values('datetime').reset_index(drop=True)

# Create future target columns for next 5 hours
for i in range(1, 6):
    df[f'aqi_next_{i}hr'] = df['aqi'].shift(-i)

# Drop rows with NaN due to shifting
df.dropna(inplace=True)

# Features and targets
X = df[['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]
y = df[[f'aqi_next_{i}hr' for i in range(1, 6)]]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("✅ MSE:", mean_squared_error(y_test, y_pred, multioutput='raw_values'))
print("✅ R2:", r2_score(y_test, y_pred, multioutput='raw_values'))

# Save model and scaler
with open('aqi_multi_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler_multi.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model & Scaler saved successfully!")
