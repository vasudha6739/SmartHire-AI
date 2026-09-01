import re


def extract_skills(resume_text):
    skills = [
        "python", "java", "c", "c++", "sql",
        "html", "css", "javascript", "react",
        "flask", "django", "pandas", "numpy",
        "machine learning", "deep learning",
        "tensorflow", "scikit-learn", "nlp",
        "aws", "azure", "docker", "git",
        "powerbi", "excel", "statistics"
    ]

    text = resume_text.lower()

    found_skills = []

    for skill in skills:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            found_skills.append(skill)

    return found_skills


def skill_gap_report(resume_text, required_skills):
    resume_skills = extract_skills(resume_text)

    required_skills = [skill.lower() for skill in required_skills]

    missing_skills = [
        skill for skill in required_skills
        if skill not in resume_skills
    ]

    matched_skills = [
        skill for skill in required_skills
        if skill in resume_skills
    ]

    return resume_skills, matched_skills, missing_skills


# Sample resume
sample_resume = """
I know Python, SQL, Pandas, NumPy, HTML and CSS.
I have worked on machine learning projects.
"""

# Skills required for a Data Scientist role
required = [
    "Python",
    "SQL",
    "Pandas",
    "NumPy",
    "Machine Learning",
    "Deep Learning",
    "TensorFlow",
    "Statistics"
]

resume_skills, matched, missing = skill_gap_report(
    sample_resume,
    required
)

print("\nSkills Found:")
print(resume_skills)

print("\nMatched Skills:")
print(matched)

print("\nSkill Gap / Missing Skills:")
print(missing)