import pandas as pd

df = pd.read_csv("creditcard_cleaned.csv")
df.sample(10000, random_state=42).to_csv("data/reference_data.csv", index=False)