"""
Drift Detector

This script compares feature distributions between a reference dataset and new data
using the two-sample Kolmogorov-Smirnov test. It writes a JSON result indicating
whether any feature shows a p-value below the configured threshold.
"""
import os
import json
import sqlite3
import pandas as pd
from scipy.stats import ks_2samp

# --- Configuration ---
# Path to the reference data (SQLite DB)
REFERENCE_DB = os.environ.get("OLD_DATA_DB", "data/reference_data.db")
# Path to the new data (SQLite DB)
NEW_DATA_DB = os.environ.get("NEW_DATA_DB", "data/requests.db")
# Drift configuration file (JSON)
DRIFT_CONFIG = os.environ.get("DRIFT_CONFIG", "drift_config.json")

# Load drift settings: threshold and feature list
with open(DRIFT_CONFIG) as cfg_file:
    config = json.load(cfg_file)
THRESHOLD = config.get("ks_pvalue_threshold", 0.05)
FEATURES = config.get("features", [])

# --- Helper functions ---
def load_from_db(path: str) -> pd.DataFrame:
    """
    Load specified features from the 'requests' table in the given SQLite database.
    """
    conn = sqlite3.connect(path)
    query = f"SELECT {','.join(FEATURES)} FROM requests"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# --- Load datasets ---
old_df = load_from_db(REFERENCE_DB)
new_df = load_from_db(NEW_DATA_DB)

# --- Perform KS-test per feature ---
drifts = {}
for feature in FEATURES:
    stat, p_value = ks_2samp(old_df[feature], new_df[feature])
    drifts[feature] = p_value

# Check if any feature's p-value is below the threshold
drift_detected = any(p < THRESHOLD for p in drifts.values())

# Write result JSON
result = {
    "drift_detected": drift_detected,
    "feature_pvalues": drifts
}
with open("drift_result.json", "w") as outfile:
    json.dump(result, outfile)

# Console output for logs
print(f"Drift detected? {drift_detected}")
