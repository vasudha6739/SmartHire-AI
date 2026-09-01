import pandas as pd

df = pd.read_csv("data/resume.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())