
import pandas as pd
import os

# -------------------------------
# CONFIG
# -------------------------------
commodity = "tomato"   # change to "tomato"

folder_path = "data/raw"

files = [
    f for f in os.listdir(folder_path)
    if f.endswith(".csv") and commodity in f.lower()
]

print(f"\n📂 Processing {commodity.upper()} files:", files)

if not files:
    raise ValueError("No CSV files found!")

df_list = []

# -------------------------------
# LOAD DATA
# -------------------------------
for file in files:
    file_path = os.path.join(folder_path, file)

    temp_df = pd.read_csv(
        file_path,
        skiprows=1,
        thousands=','
    )

    temp_df = temp_df[[
        'State', 'District', 'Market',
        'Price Date', 'Modal Price'
    ]]

    df_list.append(temp_df)

# -------------------------------
# COMBINE
# -------------------------------
df = pd.concat(df_list, ignore_index=True)

df = df.rename(columns={
    'Price Date': 'Date',
    'Modal Price': 'Price',
    'Market': 'Mandi'
})

# -------------------------------
# CLEAN
# -------------------------------
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

df = df.dropna()

print("\nInitial Clean Data:")
print(df.head())

# -------------------------------
# PROCESS EACH MANDI (IMPORTANT)
# -------------------------------
final_list = []

for mandi in df['Mandi'].unique():
    temp = df[df['Mandi'] == mandi].copy()

    # keep State + District
    state = temp['State'].iloc[0]
    district = temp['District'].iloc[0]

    # aggregate ONLY inside same mandi
    temp = temp.groupby('Date')['Price'].mean().reset_index()

    temp = temp.sort_values('Date')

    # -------------------------------
    # FILL MISSING DATES
    # -------------------------------
    temp = temp.set_index('Date')

    full_range = pd.date_range(start=temp.index.min(), end=temp.index.max())
    temp = temp.reindex(full_range)

    temp['Price'] = temp['Price'].interpolate(method='linear')

    temp = temp.reset_index()
    temp = temp.rename(columns={'index': 'Date'})

    # -------------------------------
    # FEATURES
    # -------------------------------
    temp['lag_1'] = temp['Price'].shift(1)
    temp['lag_3'] = temp['Price'].shift(3)
    temp['lag_7'] = temp['Price'].shift(7)

    temp['ma_7'] = temp['Price'].rolling(7).mean()
    temp['ma_30'] = temp['Price'].rolling(30).mean()

    temp['month'] = temp['Date'].dt.month
    temp['price_change_pct'] = temp['Price'].pct_change()
    temp['trend_7'] = temp['ma_7'] - temp['ma_30']

    # -------------------------------
    # ADD BACK LOCATION INFO
    # -------------------------------
    temp['Mandi'] = mandi
    temp['State'] = state
    temp['District'] = district

    temp = temp.dropna()

    final_list.append(temp)

# -------------------------------
# FINAL DATASET
# -------------------------------
final_df = pd.concat(final_list, ignore_index=True)

print("\nFinal Dataset:")
print(final_df.head())

print("\nColumns:")
print(final_df.columns)

# -------------------------------
# SAVE
# -------------------------------
os.makedirs("data/processed", exist_ok=True)

output_path = f"data/processed/final_{commodity}_multi_mandi.csv"
final_df.to_csv(output_path, index=False)

print(f"\n✅ Saved to {output_path}")
