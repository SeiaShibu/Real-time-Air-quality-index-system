import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('aqi_data_fetched1.csv')

X = df[['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]
y = df['aqi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=4,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100  
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
cv_mean = cv_scores.mean() * 100

print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ R² Score (Test Accuracy): {r2:.2f} or {accuracy:.2f}%")
print(f"✅ Cross-Validation Accuracy (5-Fold CV): {cv_mean:.2f}%")



plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Actual vs Predicted AQI')
plt.grid(True)
plt.show()

# 8. Save Model
with open('aqi_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model Saved Successfully!")
