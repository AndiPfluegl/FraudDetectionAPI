import pandas as pd
from scipy.stats import ks_2samp
import joblib
import os
import json

# Pfade
OLD_DATA_PATH = "data/reference_data.csv"
NEW_DATA_PATH = "data/latest_data.csv"
DRIFT_CONFIG  = "drift_config.json"

# Lade Konfiguration
with open(DRIFT_CONFIG) as f:
    config = json.load(f)
THRESHOLD = config.get("ks_pvalue_threshold", 0.05)
FEATURES  = config["features"]

# Daten laden
old = pd.read_csv(OLD_DATA_PATH)[FEATURES]
new = pd.read_csv(NEW_DATA_PATH)[FEATURES]

# KS‑Test für jede Feature‑Spalte
drifts = {}
for feat in FEATURES:
    stat, pval = ks_2samp(old[feat], new[feat])
    drifts[feat] = pval

# Ist Drift in irgendeiner Feature > Threshold?
drift_detected = any(p < THRESHOLD for p in drifts.values())

# Ergebnis speichern
result = {
    "drift_detected": drift_detected,
    "feature_pvalues": drifts
}
with open("drift_result.json", "w") as f:
    json.dump(result, f)

print("Drift detected?" , drift_detected)
