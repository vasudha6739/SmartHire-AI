import joblib

# Load model and vectorizer
model = joblib.load("model/resume_classifier.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Enter complete resume text
print("Enter resume text (type END on a new line when finished):")

resume_lines = []

while True:
    line = input()
    if line.strip().upper() == "END":
        break
    resume_lines.append(line)

resume = " ".join(resume_lines)

# Convert text
resume_tfidf = vectorizer.transform([resume])

# Predict category
prediction = model.predict(resume_tfidf)

print("\nPredicted Category:", prediction[0])