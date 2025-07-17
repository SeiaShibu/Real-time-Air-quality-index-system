import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv('data/aqi_data_fetched1.csv')

# Optional: Convert date and time to datetime
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# Sort just in case
df = df.sort_values('datetime').reset_index(drop=True)

# Create future AQI column (next hour AQI, you can do for next 5 hours)
df['aqi_next_1hr'] = df['aqi'].shift(-1)
df['aqi_next_2hr'] = df['aqi'].shift(-2)
df['aqi_next_3hr'] = df['aqi'].shift(-3)
df['aqi_next_4hr'] = df['aqi'].shift(-4)
df['aqi_next_5hr'] = df['aqi'].shift(-5)

df.dropna(inplace=True)

X = df[['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]
y = df[['aqi_next_1hr', 'aqi_next_2hr', 'aqi_next_3hr', 'aqi_next_4hr', 'aqi_next_5hr']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(f"✅ MSE:", mean_squared_error(y_test, y_pred, multioutput='raw_values'))
print(f"✅ R2:", r2_score(y_test, y_pred, multioutput='raw_values'))

# Save model and scaler
with open('aqi_multi_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler_multi.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Model & Scaler Saved for multi-hour AQI prediction!")
