from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, Response
import os
import uuid
import json
import pdfplumber  # For extracting text from PDFs
import docx  # For extracting text from DOCX files
import pandas as pd  # For university ranking CSV handling
from fuzzywuzzy import process  # For fuzzy matching universities
from sklearn.feature_extraction.text import TfidfVectorizer  # For similarity scoring
from sklearn.metrics.pairwise import cosine_similarity
import spacy  # NLP for NER (names, orgs)
import string
import re
import requests  # Used by openai SDK and potential future LLM fallbacks
import fitz  # PyMuPDF – for photo extraction from PDF
from werkzeug.utils import secure_filename
from PIL import Image  # Image processing for candidate photos
import io
import datetime
import math
import openai  # OpenAI GPT model usage
from dotenv import load_dotenv  # Load .env variables

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your_secret_key")

# Folder setup
UPLOAD_FOLDER = 'uploads'
JD_FOLDER = os.path.join(UPLOAD_FOLDER, 'jds')
CV_FOLDER = os.path.join(UPLOAD_FOLDER, 'cvs')
JD_TEXTS_FOLDER = os.path.join(UPLOAD_FOLDER, 'jd_texts')
PHOTO_FOLDER = os.path.join('static', 'images', 'candidate_photos')
os.makedirs(JD_FOLDER, exist_ok=True)
os.makedirs(CV_FOLDER, exist_ok=True)
os.makedirs(JD_TEXTS_FOLDER, exist_ok=True)
os.makedirs(PHOTO_FOLDER, exist_ok=True)
app.config['CV_FOLDER'] = CV_FOLDER

# Load models
nlp = spacy.load("en_core_web_sm")

# Helper functions
def call_openai(prompt, max_tokens=800):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI Error:", e)
        return ""

def normalize_uni_name(name):
    return name.lower().strip().translate(str.maketrans('', '', string.punctuation))

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
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

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
        return f"images/candidate_photos/{filename}"
    return None

def extract_name(text, filename):
    top = "\n".join(text.splitlines()[:20])
    doc = nlp(top)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and 2 <= len(ent.text.split()) <= 4:
            return ent.text.title()
    for line in top.splitlines():
        l = line.strip()
        if l and l[0].isalpha() and not re.search(r'[@\d:|]', l) and len(l.split()) <= 5:
            return l.title()
    base = os.path.splitext(filename)[0]
    clean = re.sub(r'(?i)(resume|cv|final|v\d+)', '', base)
    clean = re.sub(r'[_\-]+', ' ', clean).title()
    names = [w for w in clean.split() if w.istitle()]
    if names:
        return ' '.join(names[:3])
    return "Unknown"

def extract_contact(text):
    match = re.search(r'(\+?\d[\d\s\-]{8,}\d)', text)
    return match.group(1).strip() if match else "Not Found"

def parse_jd_with_llm(jd_text):
    prompt = f"""
You are an expert job description parser.
Extract the following:
1. Job Title
2. Minimum years of experience
3. 6-10 required skills, tools, or technologies
### JD:
{jd_text[:1500]}
Output format (JSON):
{{"title": "", "experience": 0, "skills": []}}
"""
    try:
        raw = call_openai(prompt)
        return json.loads(raw)
    except:
        return {"title": "", "experience": 0, "skills": []}

def extract_university(cv_text, cutoff=85):
    doc = nlp(cv_text)
    universities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    best_uni = "unknown"
    best_score = 0
    for uni in universities:
        normalized = normalize_uni_name(uni)
        match, score = process.extractOne(normalized, university_rankings.keys())
        if score >= cutoff and score > best_score:
            best_score = score
            best_uni = match
    return best_uni

def extract_skills_with_llm(cv_text):
    prompt = f"""
Extract 6-10 job-relevant skills from this resume:
{cv_text[:2000]}
Output as a Python list:
["Skill1", "Skill2", ...]
"""
    try:
        raw = call_openai(prompt)
        return json.loads(raw) if raw.startswith("[") else eval(raw)
    except:
        return []

def generate_summary_with_llm(jd_text, cv_text):
    prompt = f"""
You are a recruiter evaluating a resume against a job description.
Return 5 bullet points. Prefix with [✔ Aligned] or [✘ Not Aligned].

JD:
{jd_text[:1000]}

CV:
{cv_text[:2000]}
"""
    try:
        raw = call_openai(prompt)
        return [line.strip("-• ").strip() for line in raw.split("\n") if line.strip()][:5]
    except:
        return ["[✘ Not Aligned] Summary could not be generated."]

def is_candidate_relevant(jd_text, cv_text):
    jd_title = session.get("jd_title", "").lower()
    jd_exp = session.get("jd_experience", 0)
    jd_skills = session.get("jd_skills", [])
    prompt = f"""
Evaluate this resume strictly against the JD.
Title: {jd_title}
Min Exp: {jd_exp}
Skills: {', '.join(jd_skills)}

Resume:
{cv_text[:2000]}

Answer one of:
Relevant
Partially Relevant
Not Relevant
"""
    result = call_openai(prompt).lower()
    if "not relevant" in result:
        return "Not Relevant"
    elif "partially relevant" in result:
        return "Partially Relevant"
    elif "relevant" in result:
        return "Relevant"
    return "Not Relevant"

def similarity_score(jd_text, cv_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([jd_text, cv_text])
    return (tfidf * tfidf.T).A[0,1]

# Routes
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
        session['jd_text_filename'] = jd_text_filename
        session['jd_content'] = jd_content
        session['jd_title'] = parsed_jd.get('title', '')
        session['jd_experience'] = parsed_jd.get('experience', 0)
        session['jd_skills'] = parsed_jd.get('skills', [])
        return render_template('upload_jd_success.html',
                               jd_content=jd_content,
                               jd_title=session['jd_title'],
                               jd_experience=session['jd_experience'],
                               jd_skills=session['jd_skills'])
    return render_template('upload_jd.html')

@app.route('/upload_cvs', methods=['GET', 'POST'])
def upload_cvs():
    if request.method == 'POST':
        files = request.files.getlist('cvs')
        if not files:
            return "No CV files selected for upload.", 400
        jd_content = session.get('jd_content', '')
        max_rank = max(university_rankings.values()) if university_rankings else 1000
        candidates = []
        for file in files:
            if file and file.filename != '':
                filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                filepath = os.path.join(CV_FOLDER, filename)
                file.save(filepath)
                text = extract_text_from_pdf(filepath) if filename.endswith('.pdf') else extract_text_from_docx(filepath)
                name = extract_name(text, file.filename)
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '', name.replace(' ', '_').lower())
                photo_path = extract_photo_from_pdf(filepath, safe_name)
                university = extract_university(text)
                uni_rank = university_rankings.get(normalize_uni_name(university), 1000)
                sim_score = similarity_score(jd_content, text)
                relevance = is_candidate_relevant(jd_content, text)
                rank_score = 1 - (float(uni_rank) / max_rank)
                if relevance == "Relevant":
                    if sim_score > 0.6 and rank_score > 0.5:
                        final_score = 0.7 * sim_score + 0.3 * rank_score
                    elif sim_score > 0.5:
                        final_score = 0.6 * sim_score + 0.2 * rank_score
                    else:
                        relevance = "Partially Relevant"
                        final_score = 0.3 * sim_score + 0.1 * rank_score
                elif relevance == "Partially Relevant":
                    final_score = 0.4 * sim_score + 0.2 * rank_score if sim_score > 0.4 else 0.1 * sim_score
                else:
                    final_score = 0.0
                summary = generate_summary_with_llm(jd_content, text)
                contact = extract_contact(text)
                skills = extract_skills_with_llm(text)
                candidates.append({
                    'filename': filename,
                    'name': name,
                    'university': university.title(),
                    'uni_rank': round(float(uni_rank), 1) if uni_rank != 1000 else "Unknown",
                    'similarity_score': round(float(sim_score), 3),
                    'final_score': round(float(final_score), 3),
                    'summary': summary,
                    'photo_path': photo_path,
                    'contact': contact,
                    'applied_date': datetime.datetime.now().strftime("%Y-%m-%d"),
                    'status': 'Pending Review',
                    'relevance': relevance,
                    'skills': skills
                })
        candidates.sort(key=lambda c: c['final_score'], reverse=True)
        session['candidates'] = candidates
        return redirect(url_for('results'))
    return render_template('upload_cvs.html')

@app.route('/results')
def results():
    return render_template('results.html',
                           jd_content=session.get('jd_content', ''),
                           candidates=session.get('candidates', []))

@app.route('/view_resume/<filename>')
def view_resume(filename):
    return send_from_directory(app.config['CV_FOLDER'], filename)

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

if __name__ == '__main__':
    app.run(debug=True)
