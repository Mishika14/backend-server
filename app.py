from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz

app = Flask(__name__)

def calculate_scores(job_description, resume_text):
    # Fuzzy Matching (for exact keywords)
    fuzzy_score = fuzz.token_set_ratio(job_description, resume_text)  # Compare keywords
    fuzzy_weight = 0.4  # 40% weight

    # Cosine Similarity (for semantic matching)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_description, resume_text])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    cosine_weight = 0.6  # 60% weight

    # Combine Scores
    total_score = (fuzzy_score * fuzzy_weight) + (cosine_sim * 100 * cosine_weight)
    return total_score

@app.route('/', methods=['GET'])
def home():
    return "<p>Hello to my App</p>"

@app.route('/rank_resumes', methods=['POST'])
def rank_resumes():
    # Step 1: Get JSON data from the request
    data = request.json
    job_description = data.get('job_description', '')
    resumes = data.get('resumes', [])

    if not job_description or not resumes:
        return jsonify({"error": "Both 'job_description' and 'resumes' are required."}), 400

    # Step 2: Calculate scores for each resume
    ranked_resumes = []
    for resume in resumes:
        # Extract the resume text (combine relevant fields)
        resume_text = (
            f"{resume.get('name', '')} "
            f"{resume.get('portfolio', '')} "
            f"{resume.get('email', '')} "
            f"{resume.get('mobile', '')} "
            f"{resume.get('github', '')} "
            f"{resume.get('linkedin', '')} "
            f"{' '.join(resume.get('technical_skills', {}).get('programming_languages', []))} "
            f"{' '.join(resume.get('technical_skills', {}).get('frameworks_libraries', []))} "
            f"{' '.join(resume.get('technical_skills', {}).get('tools_platforms', []))} "
            f"{' '.join(resume.get('technical_skills', {}).get('technologies', []))} "
            f"{' '.join(resume.get('technical_skills', {}).get('soft_skills', []))} "
            f"{' '.join([proj.get('description', '') for proj in resume.get('projects', [])])} "
            f"{' '.join([exp.get('description', '') for exp in resume.get('experience', [])])} "
            f"{' '.join([pub.get('description', '') for pub in resume.get('publications', [])])} "
            f"{' '.join([act.get('description', '') for act in resume.get('extra_curricular_activities', [])])}"
        )

        # Calculate the total score
        total_score = calculate_scores(job_description, resume_text)

        # Add the resume name and score to the ranked list
        ranked_resumes.append({
            "name": resume.get("name", "Unnamed Resume"),
            "percentage_match": round(total_score, 2)  # Round to 2 decimal places
        })

    # Step 3: Rank resumes by percentage_match (descending order)
    ranked_resumes.sort(key=lambda x: x['percentage_match'], reverse=True)

    # Step 4: Return the ranked list as JSON
    return jsonify({
        "job_description": job_description,
        "ranked_resumes": ranked_resumes
    })

if __name__ == '__main__':
    app.run(debug=True)