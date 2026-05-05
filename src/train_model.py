
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib
import os
import matplotlib.pyplot as plt

# -------------------------------
# ⚙️ CONFIG
# -------------------------------
commodity = "onion"   # change to "tomato"

file_path = f"data/processed/final_{commodity}_multi_mandi.csv"

# -------------------------------
# STEP 1: Load data
# -------------------------------
df = pd.read_csv(file_path)

print(f"\n📂 Loaded dataset for: {commodity.upper()}")

# -------------------------------
# STEP 2: Target (3-day ahead)
# -------------------------------
df['target'] = df['Price'].shift(-3)
df = df.dropna()

# -------------------------------
# 🔥 STEP 3: Encode Mandi
# -------------------------------
df = pd.get_dummies(df, columns=['Mandi'])

# -------------------------------
# STEP 4: Features
# -------------------------------
base_features = [
    'lag_1', 'lag_3', 'lag_7',
    'ma_7', 'ma_30',
    'month',
    'price_change_pct',
    'trend_7'
]

mandi_features = [col for col in df.columns if col.startswith("Mandi_")]

features = base_features + mandi_features

X = df[features]
y = df['target']

print("\n📊 Total Features:", len(features))

# -------------------------------
# STEP 5: Time Series Split
# -------------------------------
tscv = TimeSeriesSplit(n_splits=5)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

errors = []

print("\n🔍 Training with TimeSeriesSplit...\n")

for i, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    error = mean_absolute_error(y_test, preds)
    errors.append(error)

    print(f"Split {i+1} MAE: {error:.2f}")

# -------------------------------
# STEP 6: Final Training
# -------------------------------
model.fit(X, y)

# -------------------------------
# STEP 7: Save Model
# -------------------------------
os.makedirs("data/processed", exist_ok=True)

model_path = f"data/processed/{commodity}_multi_mandi_model.pkl"
joblib.dump(model, model_path)

print("\n📊 Final Results:")
print("MAE for each split:", errors)
print("Average MAE:", sum(errors) / len(errors))

print(f"\n✅ Model saved at: {model_path}")

# -------------------------------
# 🔥 STEP 8: Feature Importance
# -------------------------------
importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n🔥 Feature Importance:")
print(importance_df.head(10))

# Save importance
importance_df.to_csv(
    f"data/processed/{commodity}_multi_mandi_feature_importance.csv",
    index=False
)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.gca().invert_yaxis()
plt.title(f"{commodity.upper()} Feature Importance (Top 15)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(
    f"data/processed/{commodity}_multi_mandi_feature_importance.png"
)

print("\n📈 Feature importance saved.")

# -------------------------------
# 🔥 STEP 9: Save feature list (IMPORTANT)
# -------------------------------
joblib.dump(features, f"data/processed/{commodity}_feature_columns.pkl")

print("\n💾 Feature columns saved (important for app).")
