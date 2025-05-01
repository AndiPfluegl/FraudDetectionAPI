import pandas as pd
import numpy as np
import os

# Pfade anpassen
REFERENCE_PATH = 'data/reference_data.csv'
OUTPUT_DIR    = 'data/monthly'

# Sicherstellen, dass das Ausgabeverzeichnis existiert
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Referenzdaten laden
df_ref = pd.read_csv(REFERENCE_PATH)

# Features, auf die wir Drift anwenden
FEATURE_COLS = [c for c in df_ref.columns if c.startswith('V')] + ['Amount_scaled']

# Erzeuge für 12 Monate jeweils einen driftenden Datensatz
for month in range(1, 13):
    df_new = df_ref.copy()
    # Drift auf Amount_scaled: monatlich um 0.1 erhöhen
    df_new['Amount_scaled'] += 0.1 * month

    # Drift auf V-Features: steigende Rauschanteile
    noise_scale = 0.01 * month
    for col in FEATURE_COLS:
        if col != 'Amount_scaled':
            df_new[col] += np.random.normal(loc=0, scale=noise_scale, size=len(df_new))

    # Zufällig mischen
    df_new = df_new.sample(frac=1, random_state=42)

    # Speichern
    path = os.path.join(OUTPUT_DIR, f'new_data_month_{month}.csv')
    df_new.to_csv(path, index=False)
    print(f"Monat {month:2d} → {path}")

print("Done: 12 monatliche Datensätze erzeugt.")
