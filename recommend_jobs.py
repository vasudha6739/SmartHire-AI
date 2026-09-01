import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load job dataset
jobs = pd.read_csv("jobs.csv")

# Combine job information
jobs["job_text"] = (
    jobs["Job_Title"] + " " +
    jobs["Skills"] + " " +
    jobs["Description"]
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
job_vectors = vectorizer.fit_transform(jobs["job_text"])


def recommend_jobs(resume_text, top_n=5):
    # Convert resume into vector
    resume_vector = vectorizer.transform([resume_text])

    # Calculate similarity
    similarity = cosine_similarity(resume_vector, job_vectors)[0]

    # Get top matching jobs
    top_indices = similarity.argsort()[-top_n:][::-1]

    recommendations = jobs.iloc[top_indices].copy()
    recommendations["Match_Score"] = (
        similarity[top_indices] * 100
    ).round(2)

    return recommendations[
        ["Job_Title", "Skills", "Match_Score"]
    ]


# Test with a sample resume
sample_resume = """
Python SQL Pandas NumPy Machine Learning
data analysis and machine learning projects
"""

results = recommend_jobs(sample_resume)

print("\nRecommended Jobs:")
print(results.to_string(index=False))