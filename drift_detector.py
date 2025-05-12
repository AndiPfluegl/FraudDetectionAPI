import os
import pandas as pd
from scipy.stats import ks_2samp
import sqlite3
import json

# Unterstütze beide ENV-Namen für Referenz und Neu
OLD_DATA_PATH = (
    os.environ.get("OLD_DATA_PATH")
    or os.environ.get("OLD_DATA_DB")
    or "data/reference_data.csv"
)
NEW_DATA_PATH = (
    os.environ.get("NEW_DATA_PATH")
    or os.environ.get("NEW_DATA_DB")
    or "data/requests.db"
)
DRIFT_CONFIG  = os.environ.get("DRIFT_CONFIG", "drift_config.json")

# Konfiguration
with open(DRIFT_CONFIG) as f:
    config = json.load(f)
THRESHOLD = config.get("ks_pvalue_threshold", 0.05)
FEATURES  = config["features"]

def load_from_csv(path):
    return pd.read_csv(path, usecols=FEATURES)

def load_from_db(path):
    conn = sqlite3.connect(path)
    df = pd.read_sql_query(f"SELECT {','.join(FEATURES)} FROM requests", conn)
    conn.close()
    return df

# Laden je nachdem, ob die Pfade auf .csv enden
old = load_from_csv(OLD_DATA_PATH) if OLD_DATA_PATH.lower().endswith(".csv") else load_from_db(OLD_DATA_PATH)
new = load_from_csv(NEW_DATA_PATH) if NEW_DATA_PATH.lower().endswith(".csv") else load_from_db(NEW_DATA_PATH)

# KS-Test
drifts = {}
for feat in FEATURES:
    stat, pval = ks_2samp(old[feat], new[feat])
    drifts[feat] = pval

drift_detected = any(p < THRESHOLD for p in drifts.values())

with open("drift_result.json", "w") as f:
    json.dump({"drift_detected": drift_detected, "feature_pvalues": drifts}, f)

print("Drift detected?", drift_detected)
