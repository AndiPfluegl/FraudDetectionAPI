import pandas as pd
import sqlite3

# Pfade anpassen
CSV_PATH = "data/monthly/new_data_month_12.csv"
DB_PATH  = "data/requests.db"

# CSV einlesen
df = pd.read_csv(CSV_PATH)

# Nur die Spalten nehmen, die in requests existieren
# (V1…V28, Amount_scaled, score optional, Class optional)
allowed = set(pd.read_sql_query("PRAGMA table_info(requests);", sqlite3.connect(DB_PATH))["name"])
df = df[[c for c in df.columns if c in allowed]]

# Mit der gleichen Verbindung in die Tabelle append’en
conn = sqlite3.connect(DB_PATH)
df.to_sql("requests", conn, if_exists="append", index=False)
conn.close()
print(f"{len(df)} Zeilen aus {CSV_PATH} in requests.db eingefügt.")
