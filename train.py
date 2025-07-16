import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

# Load Data
df = pd.read_csv('data/aqi_data_fetched1.csv')

X = df[['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]
y = df['aqi']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling because you scale in Streamlit
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Model
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=4,
    random_state=42
)

model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100
cv_mean = cross_val_score(model, X_train_s, y_train, cv=5).mean() * 100

print(f"✅ MSE: {mse:.2f}")
print(f"✅ R² Score: {r2:.2f} ({accuracy:.2f}%)")
print(f"✅ Cross-Validation Score: {cv_mean:.2f}%")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color='green', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.grid()
plt.show()

# Save model & scaler
with open('aqi_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Model & Scaler Saved!")
