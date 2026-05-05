# 🌾 Agri Price Intelligence System (Multi-Mandi AI)

An AI-powered decision support system that predicts agricultural commodity prices and helps farmers/traders make smarter decisions (SELL / WAIT / HOLD).

---

## 🚀 Overview

This project uses **time-series machine learning + multi-mandi data** to forecast prices of commodities like:

* 🧅 Onion
* 🍅 Tomato

It provides:

* 📊 Price prediction (3-day ahead)
* 📈 Trend analysis
* ⚠️ Risk estimation (volatility)
* 🧠 Smart decision recommendations

---

## 🎯 Problem Statement

Agricultural markets are **highly volatile**, and farmers often lack:

* Future price visibility
* Market trend understanding
* Data-driven decision support

This system solves that using **AI + real market data (Agmarknet)**.

---

## 🧠 Key Features

### 🔹 Multi-Mandi Intelligence

* Handles multiple markets (APMCs)
* Learns location-specific pricing patterns

### 🔹 Time-Series Forecasting

* Predicts prices 3 days ahead
* Uses lag, moving averages, and trend signals

### 🔹 Smart Decision Engine

* SELL / WAIT / HOLD recommendations
* Based on prediction + volatility + trend

### 🔹 Interactive Dashboard (Streamlit)

* Clean UI
* Market selection (State → District → Mandi)
* Real-time predictions & charts

---

## 🏗️ Project Architecture

```
data/
 ├── raw/              # Agmarknet data
 ├── processed/        # Cleaned + ML-ready data

src/
 ├── data_prep.py      # Data cleaning + feature engineering
 ├── train_model.py    # Model training (XGBoost)
 ├── evaluate_model.py # Model evaluation
 ├── decision_logic.py # Recommendation engine
 ├── predict.py        # CLI prediction demo

app.py                 # Streamlit web app
```

---

## ⚙️ Tech Stack

* **Python**
* **Pandas / NumPy**
* **XGBoost (ML model)**
* **Scikit-learn**
* **Streamlit (UI)**
* **Plotly (visualization)**

---

## 📊 Machine Learning Approach

### 🔹 Feature Engineering

* Lag Features → `lag_1`, `lag_3`, `lag_7`
* Moving Averages → `ma_7`, `ma_30`
* Seasonality → `month`
* Momentum → `price_change_pct`
* Trend → `trend_7`
* Location Encoding → One-hot encoded mandis

---

### 🔹 Model

* **XGBoost Regressor**
* Handles:

  * Non-linear patterns
  * Market volatility
  * Multi-feature interactions

---

### 🔹 Training Strategy

* **TimeSeriesSplit (5 splits)**
  → Prevents data leakage
  → Trains only on past, tests on future

---

## 📈 Model Performance

| Metric | Value |
| ------ | ----- |
| MAE    | ₹121  |
| RMSE   | ₹239  |
| MAPE   | ~7.2% |

👉 Interpretation:

* Avg error ≈ ₹120
* Accuracy ≈ **92–93%**

---

## 📉 Evaluation Visualization

* Actual vs Predicted comparison
* Shows model tracks real trends effectively

---

## 🖥️ Application UI

Features:

* Commodity selection
* Smart location filtering
* Real-time metrics:

  * Current price
  * Predicted price
  * Volatility
  * Trend
* Forecast graph (7 days)
* Decision output

---

## ▶️ How to Run

### 1️⃣ Clone repo

```bash
git clone https://github.com/YashvanthG/agri-price-decision-support-system.git
cd agri-price-decision-support-system
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run app

```bash
streamlit run app.py
```

---

## 📂 Data Source

* Government Agmarknet portal
* Daily price & arrival data

---

## 🔮 Future Improvements

* 🔹 Add LSTM / Deep Learning models
* 🔹 Real-time API integration
* 🔹 Weather + supply data
* 🔹 Mobile app version
* 🔹 LLM-based advisory (AI assistant)

---

## 👨‍💻 Author

**Yashvanth G**
B.Tech CSE @ LPU

* Full Stack Developer
* ML Enthusiast
* Focus: AI-driven real-world systems

---

## ⭐ If you like this project

Give it a star ⭐ and share feedback!

---
