
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import numpy as np

from src.decision_logic import generate_decision

st.set_page_config(layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data(commodity):
    df = pd.read_csv(f"data/processed/final_{commodity}_multi_mandi.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_model(commodity):
    model = joblib.load(f"data/processed/{commodity}_multi_mandi_model.pkl")
    feature_cols = joblib.load(f"data/processed/{commodity}_feature_columns.pkl")
    importance_df = pd.read_csv(f"data/processed/{commodity}_multi_mandi_feature_importance.csv")
    return model, feature_cols, importance_df


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🌾 Controls")

commodity = st.sidebar.selectbox("Select Commodity", ["onion", "tomato"])

df = load_data(commodity)
model, feature_cols, importance_df = load_model(commodity)

states = sorted(df['State'].unique())
selected_state = st.sidebar.selectbox("State", states)

districts = sorted(df[df['State'] == selected_state]['District'].unique())
selected_district = st.sidebar.selectbox("District", districts)

mandis = sorted(
    df[
        (df['State'] == selected_state) &
        (df['District'] == selected_district)
    ]['Mandi'].unique()
)

selected_mandi = st.sidebar.selectbox("Mandi", mandis)

df_mandi = df[
    (df['State'] == selected_state) &
    (df['District'] == selected_district) &
    (df['Mandi'] == selected_mandi)
].copy()

if df_mandi.empty:
    st.error("No data available.")
    st.stop()

df_mandi = df_mandi.sort_values("Date")

# -----------------------------
# CURRENT STATE
# -----------------------------
latest = df_mandi.iloc[-1]
current_price = latest['Price']

base_features = [
    'lag_1', 'lag_3', 'lag_7',
    'ma_7', 'ma_30',
    'month',
    'price_change_pct',
    'trend_7'
]

def build_input(row):
    input_dict = {col: row[col] for col in base_features}

    for col in feature_cols:
        if col.startswith("Mandi_"):
            input_dict[col] = 1 if col == f"Mandi_{selected_mandi}" else 0

    return pd.DataFrame([input_dict])[feature_cols]

input_df = build_input(latest)
predicted_price = model.predict(input_df)[0]

# -----------------------------
# METRICS
# -----------------------------
price_diff = predicted_price - current_price
price_change_pct = (price_diff / current_price) * 100
trend_strength = latest['trend_7']

trend_symbol = "➡️ Stable"
if trend_strength > 0:
    trend_symbol = "📈 Uptrend"
elif trend_strength < 0:
    trend_symbol = "📉 Downtrend"

volatility = df_mandi['Price'].rolling(7).std().iloc[-1] / current_price

# -----------------------------
# DECISION
# -----------------------------
decision_text, reason = generate_decision(
    current_price,
    predicted_price,
    volatility,
    trend_strength
)

# -----------------------------
# FORECAST
# -----------------------------
future_days = 7
future_data = []
temp_df = df_mandi.copy()

for _ in range(future_days):
    last = temp_df.iloc[-1]

    input_df = build_input(last)
    pred = model.predict(input_df)[0]

    new_date = last['Date'] + pd.Timedelta(days=1)

    temp_df = pd.concat([temp_df, pd.DataFrame([{
        'Date': new_date,
        'Price': pred
    }])], ignore_index=True)

    temp_df['lag_1'] = temp_df['Price'].shift(1)
    temp_df['lag_3'] = temp_df['Price'].shift(3)
    temp_df['lag_7'] = temp_df['Price'].shift(7)
    temp_df['ma_7'] = temp_df['Price'].rolling(7).mean()
    temp_df['ma_30'] = temp_df['Price'].rolling(30).mean()
    temp_df['month'] = temp_df['Date'].dt.month
    temp_df['price_change_pct'] = temp_df['Price'].pct_change()
    temp_df['trend_7'] = temp_df['ma_7'] - temp_df['ma_30']

    temp_df = temp_df.dropna()

    future_data.append((new_date, pred))

future_df = pd.DataFrame(future_data, columns=["Date", "Price"])

upper = future_df['Price'] * (1 + volatility)
lower = future_df['Price'] * (1 - volatility)

# -----------------------------
# UI
# -----------------------------
st.title("🌾 Agri Price Intelligence System")
st.caption(f"{selected_state} → {selected_district} → {selected_mandi}")

# Metrics
m1, m2, m3, m4 = st.columns(4)

m1.metric("Current Price", f"₹{current_price:.0f}")
m2.metric("Predicted", f"₹{predicted_price:.0f}", f"{price_change_pct:.2f}%")
m3.metric("Volatility", f"{volatility:.2%}")
m4.metric("Trend", trend_symbol)

st.markdown("---")

# Decision
st.subheader("📊 Recommendation")

if "SELL" in decision_text:
    st.error(f"🔴 {decision_text}")
elif "WAIT" in decision_text:
    st.warning(f"🟡 {decision_text}")
else:
    st.success(f"🟢 {decision_text}")

st.caption(reason)

st.markdown("---")

# -----------------------------
# FORECAST CHART
# -----------------------------
st.subheader("📈 Price Trend & Forecast")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_mandi.tail(90)['Date'],
    y=df_mandi.tail(90)['Price'],
    name="Actual",
    line=dict(width=3)
))

fig.add_trace(go.Scatter(
    x=future_df['Date'],
    y=future_df['Price'],
    name="Forecast",
    line=dict(dash="dash", width=3)
))

fig.add_trace(go.Scatter(
    x=future_df['Date'],
    y=upper,
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=future_df['Date'],
    y=lower,
    fill='tonexty',
    name="Confidence",
    line=dict(width=0)
))

fig.update_layout(height=600)

st.plotly_chart(fig, width="stretch")

st.markdown("---")

# -----------------------------
# ACTUAL vs PREDICTED (🔥 NEW)
# -----------------------------
st.subheader("📊 Model Validation")

df_eval = df_mandi.copy()
df_eval['target'] = df_eval['Price'].shift(-3)
df_eval = df_eval.dropna()

preds = []
actuals = []
dates = []

for _, row in df_eval.tail(120).iterrows():
    input_df = build_input(row)
    pred = model.predict(input_df)[0]

    preds.append(pred)
    actuals.append(row['target'])
    dates.append(row['Date'])

fig_val = go.Figure()

fig_val.add_trace(go.Scatter(
    x=dates,
    y=actuals,
    name="Actual",
    line=dict(width=3)
))

fig_val.add_trace(go.Scatter(
    x=dates,
    y=preds,
    name="Predicted",
    line=dict(dash="dash")
))

fig_val.update_layout(height=500)

st.plotly_chart(fig_val, width="stretch")

st.markdown("---")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("🧠 Key Drivers")

clean_imp = importance_df[~importance_df['Feature'].str.contains("Mandi_")]
top_features = clean_imp.head(10)

fig_imp = go.Figure()

fig_imp.add_trace(go.Bar(
    x=top_features['Importance'],
    y=top_features['Feature'],
    orientation='h'
))

fig_imp.update_layout(height=400, yaxis=dict(autorange="reversed"))

st.plotly_chart(fig_imp, width="stretch")
