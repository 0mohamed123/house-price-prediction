import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import joblib
import os

# Load dataset
df = pd.read_csv("Data/AmesHousing.csv")

# Add engineered feature: HouseAge
df["HouseAge"] = 2025 - df["Year Built"]

# Select more features
features = [
    "Lot Area", "Overall Qual", "Overall Cond", "Gr Liv Area",
    "Total Bsmt SF", "Garage Cars", "Full Bath", "Bedroom AbvGr",
    "Kitchen Qual", "HouseAge"
]

# Handle categorical (Kitchen Qual)
df["Kitchen Qual"] = df["Kitchen Qual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1})

X = df[features].fillna(0)  # fill missing with 0
y = df["SalePrice"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (XGBoost)
model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"✅ Model trained with XGBoost! MAE: {mae}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/house_price_model.pkl")
print("💾 Model saved in models/house_price_model.pkl")