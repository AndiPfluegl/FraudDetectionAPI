import os
import pandas as pd
from scipy.stats import ks_2samp
import sqlite3
import json

# Pfade per ENV (CSV für Referenz, SQLite für neue Daten)
OLD_DATA_PATH = os.environ.get("OLD_DATA_PATH", "data/reference_data.csv")
NEW_DATA_DB   = os.environ.get("NEW_DATA_DB",   "data/requests.db")
DRIFT_CONFIG  = os.environ.get("DRIFT_CONFIG",  "drift_config.json")

# Lade Konfiguration für Threshold und Features
with open(DRIFT_CONFIG) as f:
    config = json.load(f)
THRESHOLD = config.get("ks_pvalue_threshold", 0.05)
FEATURES  = config["features"]

# Hilfsfunktion: Alte Daten aus CSV laden
def load_from_csv(path):
    df = pd.read_csv(path, usecols=FEATURES)
    return df

# Hilfsfunktion: Neue Daten aus SQLite-DB laden
def load_from_db(path):
    conn = sqlite3.connect(path)
    df = pd.read_sql_query(f"SELECT {','.join(FEATURES)} FROM requests", conn)
    conn.close()
    return df

# Alte und neue Daten einlesen
if OLD_DATA_PATH.lower().endswith('.csv'):
    old = load_from_csv(OLD_DATA_PATH)
else:
    old = load_from_db(OLD_DATA_PATH)
new = load_from_db(NEW_DATA_DB)

# KS-Test für jede Feature-Spalte
drifts = {}
for feat in FEATURES:
    stat, pval = ks_2samp(old[feat], new[feat])
    drifts[feat] = pval

# Ist Drift erkannt? (p < Threshold)
drift_detected = any(p < THRESHOLD for p in drifts.values())

# Ergebnis speichern
result = {
    "drift_detected": drift_detected,
    "feature_pvalues": drifts
}
with open("drift_result.json", "w") as f:
    json.dump(result, f)

print("Drift detected?", drift_detected)
