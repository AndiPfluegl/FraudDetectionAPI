import os
import pandas as pd
import numpy as np
import subprocess

# Pfade anpassen
REFERENCE_PATH = 'data/reference_data.csv'
OUTPUT_DIR = 'data/monthly'

# Stelle sicher, dass das Ausgabeverzeichnis existiert
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Referenzdaten laden
df_ref = pd.read_csv(REFERENCE_PATH)

# Features, auf die wir Drift anwenden
# Hier alle 'V'-Spalten und 'Amount_scaled'
FEATURE_COLS = [c for c in df_ref.columns if c.startswith('V')] + ['Amount_scaled']

print("Erstelle simulierte Monatsdaten mit zunehmendem Drift…")
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
    print(f"  Monat {month:2d} → {path}")

print("\nJetzt kannst du für jeden Monat das drift_detector.py aufrufen:")
print("Beispiel für Monat 1:")
print(f"  NEW_DATA_PATH={OUTPUT_DIR}/new_data_month_1.csv python drift_detector.py")

# Optional: automatisch alle Monate durchtesten
run_all = input("\nMöchtest du jetzt direkt alle 12 Monate testen? (j/N) ")
if run_all.lower() == 'j':
    for month in range(1, 13):
        new_path = f"{OUTPUT_DIR}/new_data_month_{month}.csv"
        print(f"\n--- Test Monat {month} ---")
        env = os.environ.copy()
        env['NEW_DATA_PATH'] = new_path
        # Hier muss drift_detector.py so programmiert sein, dass es NEW_DATA_PATH liest
        subprocess.run(['python', 'drift_detector.py'], env=env)
