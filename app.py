import pandas as pd
import joblib
from flask import Flask, render_template, request

from recommend_jobs import recommend_jobs
from skill_gap import skill_gap_report

# Flask app
# Your index.html is currently in the main SmartHire folder
app = Flask(__name__, template_folder=".")

# Load trained classifier and TF-IDF vectorizer
classifier = joblib.load("model/resume_classifier.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Load jobs
jobs = pd.read_csv("jobs.csv")


@app.route("/", methods=["GET", "POST"])
def home():

    predicted_category = None
    recommendations = None
    matched_skills = []
    missing_skills = []
    resume_text = ""

    if request.method == "POST":

        resume_text = request.form.get("resume", "")

        if resume_text.strip():

            # -------------------------------
            # TASK 1: RESUME CLASSIFIER
            # -------------------------------
            resume_vector = vectorizer.transform([resume_text])
            predicted_category = classifier.predict(resume_vector)[0]

            # -------------------------------
            # TASK 2: JOB RECOMMENDER
            # -------------------------------
            recommendations = recommend_jobs(
                resume_text,
                top_n=5
            )

            # -------------------------------
            # TASK 3: SKILL GAP REPORT
            # -------------------------------
            required_skills = [
                "Python",
                "SQL",
                "Pandas",
                "NumPy",
                "Machine Learning",
                "Deep Learning",
                "TensorFlow",
                "Statistics"
            ]

            _, matched_skills, missing_skills = skill_gap_report(
                resume_text,
                required_skills
            )

    return render_template(
        "index.html",
        predicted_category=predicted_category,
        recommendations=recommendations,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        resume_text=resume_text
    )


if __name__ == "__main__":
    app.run(debug=True)