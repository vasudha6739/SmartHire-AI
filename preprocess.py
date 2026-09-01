import pandas as pd
import re

# Load dataset
df = pd.read_csv("data/resume.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("Columns found:", df.columns.tolist())

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df = df.dropna(subset=["Resume_str", "Category"])

df["clean_resume"] = df["Resume_str"].apply(clean_text)

# Save processed dataset
df.to_csv("data/processed_resume.csv", index=False)

print("Preprocessing completed!")
print("Total resumes:", len(df))
print("Categories:", df["Category"].nunique())
df.to_csv("data/processed_resumes.csv", index=False)

print("Processed dataset saved successfully!")