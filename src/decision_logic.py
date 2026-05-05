
def generate_decision(current_price, predicted_price, volatility, trend_strength=None):
    change = (predicted_price - current_price) / current_price

    # -----------------------------
    # Dynamic thresholds
    # -----------------------------
    strong_up = volatility * 1.5
    mild_up = volatility * 0.5
    mild_down = -volatility * 0.5
    strong_down = -volatility * 1.5

    # -----------------------------
    # Decision Logic (UPGRADED 🔥)
    # -----------------------------

    if change >= strong_up:
        decision = "🟢 SELL NOW"
        reason = "Strong upward movement beyond expected volatility"
        confidence = "High"

    elif change >= mild_up:
        decision = "🟡 WAIT"
        reason = "Moderate upward trend, potential for further increase"
        confidence = "Medium"

    elif change <= strong_down:
        decision = "🔴 RISK – SELL EARLY"
        reason = "Sharp expected drop beyond normal fluctuation"
        confidence = "High"

    elif change <= mild_down:
        decision = "🔴 RISK"
        reason = "Slight downward pressure in price"
        confidence = "Medium"

    else:
        decision = "🟡 WAIT"
        reason = "Market is stable within expected range"
        confidence = "Low"

    # -----------------------------
    # Trend adjustment (NEW 🔥)
    # -----------------------------
    if trend_strength is not None:
        if trend_strength > 0:
            reason += " + upward trend support"
        elif trend_strength < 0:
            reason += " + downward trend pressure"

    return decision, f"{reason} (Confidence: {confidence})"

