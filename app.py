from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, Response
import os
import uuid
import json
import pdfplumber
import docx
import pandas as pd
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import string
import re
import requests
import fitz  # PyMuPDF
from werkzeug.utils import secure_filename
from PIL import Image
import io
import datetime
import math

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
JD_FOLDER = os.path.join(UPLOAD_FOLDER, 'jds')
CV_FOLDER = os.path.join(UPLOAD_FOLDER, 'cvs')
JD_TEXTS_FOLDER = os.path.join(UPLOAD_FOLDER, 'jd_texts')
PHOTO_FOLDER = os.path.join('static', 'images', 'candidate_photos')

os.makedirs(JD_FOLDER, exist_ok=True)
os.makedirs(CV_FOLDER, exist_ok=True)
os.makedirs(JD_TEXTS_FOLDER, exist_ok=True)
os.makedirs(PHOTO_FOLDER, exist_ok=True)

app.config['JD_FOLDER'] = JD_FOLDER
app.config['CV_FOLDER'] = CV_FOLDER
app.config['JD_TEXTS_FOLDER'] = JD_TEXTS_FOLDER
app.config['PHOTO_FOLDER'] = PHOTO_FOLDER

nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def normalize_uni_name(name):
    name = name.lower().strip()
    name = name.translate(str.maketrans('', '', string.punctuation))
    return name

def parse_rank(rank_str):
    if isinstance(rank_str, str) and '-' in rank_str:
        try:
            start, end = map(int, rank_str.split('-'))
            return (start + end) / 2
        except:
            return 1000
    try:
        return float(rank_str)
    except:
        return 1000

def load_university_rankings(csv_path='data/university_rankings_minimal.csv'):
    df = pd.read_csv(csv_path)
    df['institution_normalized'] = df['Institution Name'].str.lower().str.strip()
    df['institution_normalized'] = df['institution_normalized'].apply(normalize_uni_name)
    df['2024 RANK'] = df['2024 RANK'].apply(parse_rank)
    return df.set_index('institution_normalized')['2024 RANK'].to_dict()

university_rankings = load_university_rankings()

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        texts = [page.extract_text() for page in pdf.pages if page.extract_text()]
        return "\n".join(texts)

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_photo_from_pdf(pdf_path, output_name):
    doc = fitz.open(pdf_path)
    best_candidate = None
    best_score = 0
    for page in doc:
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"].lower()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                width, height = image.size
                area = width * height
                aspect_ratio = height / width if width else 0
                if not (100 < width < 800 and 100 < height < 1000):
                    continue
                if not (0.8 <= aspect_ratio <= 2.0):
                    continue
                gray = image.convert("L")
                histogram = gray.histogram()
                entropy = -sum((p / area * (0 if p == 0 else math.log2(p / area))) for p in histogram)
                if entropy < 3.0:
                    continue
                score = area * entropy
                if score > best_score:
                    best_score = score
                    best_candidate = (image, image_ext)
            except Exception as e:
                print(f"Photo extract error (page {page.number}):", e)
    if best_candidate:
        image, ext = best_candidate
        filename = f"{output_name}.{ext}"
        save_path = os.path.join(PHOTO_FOLDER, filename)
        image.save(save_path)
        return save_path
    return None

def extract_education_section(text):
    sections = re.split(r'\n\s*(education|academic background|qualifications)\s*\n', text, flags=re.IGNORECASE)
    if len(sections) > 2:
        return sections[2]
    return text

def extract_universities_spacy(cv_text):
    doc = nlp(cv_text)
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]

def extract_university_with_llm(text):
    prompt = f"Extract the name of the university where the candidate studied:\n{text}\nIf not found, return 'unknown'."
    try:
        response = requests.post("http://localhost:11434/api/generate", json={"model": "gemma:2b", "prompt": prompt, "stream": False})
        if response.ok:
            return response.json().get("response", "unknown").strip().lower()
    except Exception as e:
        print("LLM fallback failed:", e)
    return "unknown"

def parse_jd_with_llm(jd_text):
    prompt = f"""
You are an expert job description parser.

Your task is to extract the following ONLY IF they are clearly mentioned in the job description:
1. Job Title
2. Minimum years of work experience required
3. 6 to 10 required skills, tools, or technologies

⚠️ DO NOT GUESS or ADD skills based on common industry roles.
❌ DO NOT hallucinate Python, SQL, Java, AWS, etc., unless they are mentioned in the text.

### Job Description:
{jd_text[:1500]}

### Output format (strict JSON):
{{
  "title": "",
  "experience": 0,
  "skills": ["", "", ...]
}}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma:2b", "prompt": prompt, "stream": False}
        )
        if response.ok:
            raw = response.json().get("response", "").strip()
            return json.loads(raw)
    except Exception as e:
        print("LLM JD parsing failed:", e)

    return {"title": "", "experience": 0, "skills": []}


def extract_university(cv_text, cutoff=85):
    edu_section = extract_education_section(cv_text)
    edu_section_clean = normalize_uni_name(edu_section)
    for known_uni in university_rankings.keys():
        if known_uni in edu_section_clean:
            return known_uni
    universities = extract_universities_spacy(edu_section)
    best_uni = "unknown"
    best_score = 0
    for uni in universities:
        normalized = normalize_uni_name(uni)
        match, score = process.extractOne(normalized, university_rankings.keys())
        if score >= cutoff and score > best_score:
            best_score = score
            best_uni = match
    if best_uni != "unknown":
        return best_uni
    return extract_university_with_llm(cv_text)

def extract_contact(text):
    contact = "Not Found"
    phone_match = re.search(r'(\+?\d[\d\s\-]{8,}\d)', text)
    if phone_match:
        contact = phone_match.group(1).strip()
    return contact

def extract_skills_with_llm(cv_text):
    prompt = f"""
You are an expert resume analyzer.

Your job is to extract **actual technical and professional skills** from the resume below, ONLY if the candidate has demonstrated them in a real job or project (not just listed or studied).

Ignore soft skills like "communication" or "teamwork". Return 6 to 10 real, role-relevant skills.

### Resume Text:
{cv_text[:2000]}

Return output as a Python list of strings:
["Skill1", "Skill2", ...]
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma:2b", "prompt": prompt, "stream": False}
        )
        if response.ok:
            raw = response.json().get("response", "").strip()
            # Handle basic formatting cleanup
            skills = json.loads(raw) if raw.startswith("[") else eval(raw)
            return [s.strip() for s in skills if isinstance(s, str)]
    except Exception as e:
        print("LLM skill extraction failed:", e)
    
    return []


def get_embedding(text):
    return bert_model.encode([text])[0]

def similarity_score(jd_text, cv_text):
    jd_emb = get_embedding(jd_text)
    cv_emb = get_embedding(cv_text)
    return float(cosine_similarity([jd_emb], [cv_emb])[0][0])

def generate_summary_with_llm(jd_text, cv_text):
    prompt = f"""
You are a senior recruiter at a hiring agency, responsible for evaluating how well a single candidate matches a specific job description.

Objective:
Provide an unbiased, industry-agnostic evaluation based strictly on the candidate's demonstrated experience and the job’s stated requirements.

IMPORTANT RULES:
- DO NOT assume relevance based on the candidate’s industry or company.
- DO NOT infer alignment from titles alone — evaluate described duties and technologies.
- DO NOT include any domain-specific praise (e.g., telecom, banking, healthcare) unless the job description explicitly demands domain experience.
- DO NOT hallucinate skills or infer knowledge not explicitly stated in the resume.

OUTPUT REQUIREMENTS:
- Return exactly 5 bullet points.
- Each bullet must start with either:
  - [✔ Aligned] — if resume clearly demonstrates a required skill/experience.
  - [✘ Not Aligned] — if resume lacks or vaguely references a required item.
- Be brief and factual. Do not include educational info unless it is mentioned in the job requirements.

Job Description:
{jd_text[:1000]}

Candidate Resume:
{cv_text[:2000]}

Output example:
- [✔ Aligned] Developed ETL pipelines using Python and SQL.
- [✘ Not Aligned] No evidence of cloud platform experience (e.g., Azure, AWS).
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma:2b", "prompt": prompt, "stream": False}
        )
        if response.ok:
            summary = response.json().get("response", "").strip()
            summary_points = [
                re.sub(r"^[\-\u2022\d\.\)\s]+", "", line).strip()
                for line in summary.split("\n") if line.strip()
            ]
            return summary_points[:5]
    except Exception as e:
        print("Gemma summary failed:", e)
    return ["[✘ Not Aligned] Summary could not be generated."]


# UPDATED relevance function with domain mismatch logic
def is_candidate_relevant(jd_text, cv_text):
    jd_title = session.get("jd_title", "").lower()
    jd_experience = session.get("jd_experience", 0)
    jd_skills = session.get("jd_skills", [])
    skill_list = ', '.join(jd_skills)

    prompt = f"""
You are an expert AI resume evaluator.

Evaluate the following candidate **strictly** against the job description.

### JOB REQUIREMENTS:
- Title: {jd_title}
- Minimum Experience: {jd_experience} years
- Must-Have Skills: {skill_list}

### INSTRUCTIONS:
- Carefully read the resume content.
- If resume shows unrelated work experience (e.g., finance, HR, accounting) and no matching job duties, return: Not Relevant
- Do not assume someone is relevant unless technical roles and job duties are clearly present
- Vague references to Excel, data entry, or office work DO NOT count as relevant

### RESPONSE:
Return one of exactly these 3 options:
Relevant
Partially Relevant
Not Relevant

### RESUME:
{cv_text[:2000]}

Your Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma:2b", "prompt": prompt, "stream": False}
        )
        if response.ok:
            result = response.json().get("response", "").strip().lower()
            print("LLM Relevance Response:", result)

            if "not relevant" in result:
                return "Not Relevant"
            elif "partially relevant" in result:
                return "Partially Relevant"
            elif "relevant" in result:
                return "Relevant"
            else:
                print(f"⚠️ Unexpected LLM response: {result}")
                return "Not Relevant"
    except Exception as e:
        print("LLM relevance check failed:", e)

    return "Not Relevant"

# Additional helper function to add more validation
def validate_job_field_match(jd_title, cv_text):
    """Extra validation to catch obvious field mismatches"""
    jd_title_lower = jd_title.lower()
    cv_text_lower = cv_text.lower()
    
    # Define field keywords
    data_fields = ['data', 'analytics', 'database', 'sql', 'python', 'architect', 'engineer', 'scientist']
    accounting_fields = ['accounting', 'bookkeeping', 'accounts', 'ledger', 'financial', 'audit', 'tax']
    marketing_fields = ['marketing', 'advertising', 'campaign', 'brand', 'social media']
    
    # Check if JD is data-related
    if any(keyword in jd_title_lower for keyword in data_fields):
        # Check if CV is clearly from accounting/finance
        accounting_matches = sum(1 for keyword in accounting_fields if keyword in cv_text_lower)
        data_matches = sum(1 for keyword in data_fields if keyword in cv_text_lower)
        
        if accounting_matches > 3 and data_matches < 2:
            return False  # Clear mismatch
            
    return True  # No obvious mismatch detected


@app.route('/')
def index():
    return redirect(url_for('upload_jd'))

@app.route('/upload_jd', methods=['GET', 'POST'])
def upload_jd():
    if request.method == 'POST':
        text = request.form.get('jd_text')
        file = request.files.get('jd_file')

        if file and file.filename != '':
            filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
            filepath = os.path.join(JD_FOLDER, filename)
            file.save(filepath)
            jd_content = extract_text_from_pdf(filepath) if filename.endswith('.pdf') else extract_text_from_docx(filepath)
        elif text and text.strip():
            jd_content = text.strip()
        else:
            return "Please provide a job description either as text or file.", 400

        jd_text_filename = f"{uuid.uuid4()}.txt"
        jd_text_path = os.path.join(JD_TEXTS_FOLDER, jd_text_filename)
        with open(jd_text_path, 'w', encoding='utf-8') as f:
            f.write(jd_content)

        parsed_jd = parse_jd_with_llm(jd_content)

        # Store values in session
        session['jd_text_filename'] = jd_text_filename
        session['jd_content'] = jd_content
        session['jd_title'] = parsed_jd.get('title', '')
        session['jd_experience'] = parsed_jd.get('experience', 0)
        session['jd_skills'] = parsed_jd.get('skills', [])

        return render_template(
            'upload_jd_success.html',
            jd_content=jd_content,
            jd_title=session['jd_title'],
            jd_experience=session['jd_experience'],
            jd_skills=session['jd_skills']
        )

    return render_template('upload_jd.html')

@app.route('/upload_cvs', methods=['GET', 'POST'])
@app.route('/upload_cvs', methods=['GET', 'POST'])

@app.route('/upload_cvs', methods=['GET', 'POST'])
def upload_cvs():
    # Helper function to extract candidate's actual name from top lines of resume text
    def extract_name_from_cv(text):
        lines = text.strip().split("\n")[:10]  # Consider first 10 lines only
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(c in line for c in "@0123456789:|") or len(line.split()) > 5:
                continue  # Skip lines with email, numbers, or too long
            if all(word[0].isupper() for word in line.split() if word.isalpha()):
                return line.strip()
        return None  # fallback if no proper name found

    if request.method == 'POST':
        files = request.files.getlist('cvs')
        if not files:
            return "No CV files selected for upload.", 400

        jd_content = session.get('jd_content', '')
        max_rank = max(university_rankings.values()) if university_rankings else 1000
        candidates = []

        for file in files:
            if file and file.filename != '':
                # Save uploaded file
                filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                filepath = os.path.join(CV_FOLDER, filename)
                file.save(filepath)

                # Extract text from file (PDF or DOCX)
                text = extract_text_from_pdf(filepath) if filename.endswith('.pdf') else extract_text_from_docx(filepath)

                # 🆕 Extract actual candidate name from text, fallback to filename if not found
                extracted_name = extract_name_from_cv(text)
                name_part = extracted_name if extracted_name else os.path.splitext(file.filename)[0].replace('_', ' ').replace('-', ' ').title()

                # University extraction and ranking
                university = extract_university(text)
                normalized_uni = normalize_uni_name(university)
                uni_rank = university_rankings.get(normalized_uni, 1000)

                # Similarity scoring using BERT
                sim_score = similarity_score(jd_content, text)

                # Relevance classification using job title and LLM
                if not validate_job_field_match(session['jd_title'], text):
                    relevance_tag = "Not Relevant"
                else:
                    relevance_tag = is_candidate_relevant(jd_content, text)

                # Ranking score: normalized university rank
                rank_score = 1 - (float(uni_rank) / max_rank)

                # Final score logic based on relevance and similarity
                if relevance_tag == "Relevant":
                    if sim_score > 0.6 and rank_score > 0.5:
                        final_score = 0.7 * sim_score + 0.3 * rank_score
                        reason = "Strong match: high similarity + good university"
                    elif sim_score > 0.5:
                        final_score = 0.6 * sim_score + 0.2 * rank_score
                        reason = "Good match: decent similarity"
                    else:
                        relevance_tag = "Partially Relevant"
                        final_score = 0.3 * sim_score + 0.1 * rank_score
                        reason = "Downgraded: low similarity despite relevance"
                elif relevance_tag == "Partially Relevant":
                    if sim_score > 0.4:
                        final_score = 0.4 * sim_score + 0.2 * rank_score
                        reason = "Partial match with acceptable similarity"
                    else:
                        final_score = 0.1 * sim_score
                        reason = "Weak partial match"
                else:
                    final_score = 0.0
                    reason = "Not relevant to job requirements"

                print(f"[SCORING] {name_part} → {relevance_tag} | Sim: {round(sim_score,3)} | Final: {round(final_score,3)} | {reason}")

                # LLM-based summary of alignment to JD
                summary = generate_summary_with_llm(jd_content, text)

                # Extract contact details from text
                contact = extract_contact(text)

                # Extract candidate photo from CV
                photo_filename = extract_photo_from_pdf(filepath, name_part.replace(' ', '_').lower())
                if photo_filename:
                    photo_url = photo_filename.replace("\\", "/")
                    if photo_url.startswith("static/"):
                        photo_url = photo_url[len("static/"):]
                else:
                    photo_url = None

                # Extract LLM-driven contextual skills
                skills = extract_skills_with_llm(text)

                # Assemble full candidate profile
                candidates.append({
                    'filename': filename,
                    'name': name_part,
                    'text_excerpt': text[:300],
                    'university': university.title(),
                    'uni_rank': round(float(uni_rank), 1) if uni_rank != 1000 else "Unknown",
                    'similarity_score': round(float(sim_score), 3),
                    'final_score': round(float(final_score), 3),
                    'summary': summary,
                    'photo_path': photo_url,
                    'contact': contact,
                    'applied_date': datetime.datetime.now().strftime("%Y-%m-%d"),
                    'status': 'Pending Review',
                    'relevance': relevance_tag,
                    'relevance_reason': reason,
                    'skills': skills
                })

        # Sort candidates by descending final score
        candidates.sort(key=lambda c: c['final_score'], reverse=True)
        session['candidates'] = candidates
        return redirect(url_for('results'))

    return render_template('upload_cvs.html')


@app.route('/results')
def results():
    candidates = session.get('candidates', [])
    jd_content = session.get('jd_content', '')
    return render_template('results.html', jd_content=jd_content, candidates=candidates)

@app.route('/export_csv')
def export_csv():
    candidates = session.get('candidates', [])
    if not candidates:
        return "No candidates to export.", 400

    def generate():
        data = ["Name,University,University Rank,Similarity Score,Final Score,Contact,Status,Relevance,Top Skills"]
        for c in candidates:
            skills = ", ".join(c["skills"])
            row = f"{c['name']},{c['university']},{c['uni_rank']},{c['similarity_score']},{c['final_score']},{c['contact']},{c['status']},{c['relevance']},{skills}"
            data.append(row)
        return "\n".join(data)

    return Response(generate(), mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=shortlisted_candidates.csv"})

@app.route('/view_resume/<filename>')
def view_resume(filename):
    return send_from_directory(app.config['CV_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
