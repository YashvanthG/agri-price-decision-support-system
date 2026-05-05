import pandas as pd
import joblib

from src.decision_logic import generate_decision

# -------------------------------
# ⚙️ CONFIG
# -------------------------------

commodity = "onion"   # change to "tomato"

# -------------------------------
# STEP 1: Load dataset
# -------------------------------

df = pd.read_csv(f"data/processed/final_{commodity}_prices.csv")

# -------------------------------
# STEP 2: Load trained model
# -------------------------------

model = joblib.load(f"data/processed/{commodity}_model.pkl")

# -------------------------------
# STEP 3: Get latest data
# -------------------------------

latest = df.iloc[-1]

current_price = latest['Price']

features = ['Price', 'lag_1', 'lag_3', 'lag_7', 'ma_7', 'ma_30']
input_data = latest[features].values.reshape(1, -1)

# -------------------------------
# STEP 4: Predict future price
# -------------------------------

predicted_price = model.predict(input_data)[0]

# -------------------------------
# STEP 5: Decision Logic
# -------------------------------

decision = generate_decision(current_price, predicted_price)

# -------------------------------
# STEP 6: Output
# -------------------------------

print("\n===== FINAL OUTPUT =====")
print(f"Commodity: {commodity.upper()}")
print(f"Current Price: ₹{current_price:.2f}")
print(f"Predicted Price (3 days): ₹{predicted_price:.2f}")
print(f"Decision: {decision}")