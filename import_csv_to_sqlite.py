"""
Utility script to import a monthly CSV into the existing SQLite `requests` table.
It only keeps columns that match the table schema (e.g., V1..V28, Amount_scaled, score, Class).
"""

import pandas as pd
import sqlite3
import os

# --- Configuration ---
CSV_PATH = os.environ.get("CSV_PATH", "data/monthly/new_data_month_12.csv")
DB_PATH = os.environ.get("DB_PATH", "data/requests.db")

# --- Load CSV into DataFrame ---
# Read the CSV file; adjust `usecols` if you know the exact schema in advance
print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# --- Determine allowed columns from existing SQLite schema ---
print(f"Inspecting table schema in {DB_PATH}...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.execute("PRAGMA table_info(requests);")
# PRAGMA returns (cid, name, type, notnull, dflt_value, pk)
schema_info = cursor.fetchall()
conn.close()
allowed_columns = {row[1] for row in schema_info}
print(f"Allowed columns: {allowed_columns}")

# --- Filter DataFrame to allowed columns only ---
filtered_cols = [col for col in df.columns if col in allowed_columns]
df = df[filtered_cols]
print(f"Filtered DataFrame to {len(filtered_cols)} columns; shape={df.shape}")

# --- Append to SQLite `requests` table ---
print(f"Appending {len(df)} rows to {DB_PATH}...")
conn = sqlite3.connect(DB_PATH)
# Note: `if_exists='append'` requires matching schema; index=False avoids creating an extra column
df.to_sql("requests", conn, if_exists="append", index=False)
conn.close()

print(f"Successfully imported {len(df)} rows into `requests` table.")