
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# CONFIG
# -----------------------------
commodity = "onion"   # change to tomato

# -----------------------------
# LOAD
# -----------------------------
df = pd.read_csv(f"data/processed/final_{commodity}_multi_mandi.csv")
model = joblib.load(f"data/processed/{commodity}_multi_mandi_model.pkl")
feature_cols = joblib.load(f"data/processed/{commodity}_feature_columns.pkl")

df['Date'] = pd.to_datetime(df['Date'])

# -----------------------------
# TARGET
# -----------------------------
df['target'] = df['Price'].shift(-3)
df = df.dropna()

# -----------------------------
# BUILD FEATURES
# -----------------------------
X_list = []
y_list = []
dates = []

for _, row in df.iterrows():

    input_dict = {
        'lag_1': row['lag_1'],
        'lag_3': row['lag_3'],
        'lag_7': row['lag_7'],
        'ma_7': row['ma_7'],
        'ma_30': row['ma_30'],
        'month': row['month'],
        'price_change_pct': row['price_change_pct'],
        'trend_7': row['trend_7']
    }

    for col in feature_cols:
        if col.startswith("Mandi_"):
            mandi_name = col.replace("Mandi_", "")
            input_dict[col] = 1 if row['Mandi'] == mandi_name else 0

    X_list.append([input_dict[col] for col in feature_cols])
    y_list.append(row['target'])
    dates.append(row['Date'])

X = np.array(X_list)
y = np.array(y_list)
dates = np.array(dates)

# -----------------------------
# SPLIT
# -----------------------------
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test = dates[split:]

# -----------------------------
# PREDICT
# -----------------------------
preds = model.predict(X_test)

# -----------------------------
# METRICS
# -----------------------------
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

print("\n📊 MODEL EVALUATION")
print(f"MAE  : ₹{mae:.2f}")
print(f"RMSE : ₹{rmse:.2f}")
print(f"MAPE : {mape:.2f}%")

# -----------------------------
# GRAPH (IMPORTANT 🔥)
# -----------------------------
plt.figure(figsize=(12, 6))

plt.plot(dates_test[-100:], y_test[-100:], label="Actual", linewidth=2)
plt.plot(dates_test[-100:], preds[-100:], label="Predicted", linestyle='dashed')

plt.title(f"{commodity.upper()} Price: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

plt.tight_layout()
plt.savefig(f"data/processed/{commodity}_actual_vs_pred.png")

print("\n📈 Graph saved → data/processed/")
plt.show()
