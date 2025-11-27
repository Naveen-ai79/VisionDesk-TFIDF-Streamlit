import os
import re
import tempfile
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional

import streamlit as st
import pandas as pd
import numpy as np

from werkzeug.utils import secure_filename

import pdfplumber
import PyPDF2
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv

# ==========================================================
# LOAD ENV
# ==========================================================
load_dotenv()

# ==========================================================
# FILE UTILS
# ==========================================================
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc'}

def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def secure_temp_save(file_storage):
    original_name = getattr(file_storage, "filename", None) or getattr(file_storage, "name", None)
    filename = secure_filename(original_name)
    suffix = os.path.splitext(filename)[1]
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    # Streamlit file upload
    data = file_storage.getbuffer()
    with open(tmp_path, "wb") as f:
        f.write(data)

    return tmp_path

# ==========================================================
# SKILL EXTRACTION
# ==========================================================
COMMON_SKILLS = [
    "python", "django", "flask", "fastapi", "rest", "api", "sql", "postgres", "mysql",
    "aws", "azure", "gcp", "aws lambda", "azure functions", "cloud", "docker", "kubernetes",
    "machine learning", "ml", "data science", "pandas", "numpy", "spark"
]

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

def normalize_text(text: str):
    return re.sub(r'\s+', ' ', (text or "").lower()).strip()

def extract_skills_from_text(text: str):
    t = normalize_text(text)
    found = set()
    for skill in COMMON_SKILLS:
        if skill in t:
            found.add(skill)
    return list(found)

def extract_emails_from_text(text: str):
    if not text:
        return []
    return EMAIL_REGEX.findall(text)

# ==========================================================
# RESUME PARSER
# ==========================================================
def extract_text_from_pdf(path):
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text_parts.append(t)
    except Exception:
        try:
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for p in reader.pages:
                    t = p.extract_text()
                    if t:
                        text_parts.append(t)
        except Exception as e:
            raise RuntimeError(f"PDF parsing failed: {e}")
    return "\n".join(text_parts).strip()

def extract_text_from_docx(path):
    try:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception as e:
        raise RuntimeError(f"DOCX parsing failed: {e}")

def parse_resume(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type: " + ext)

# ==========================================================
# TF-IDF SERVICE (Semantic Similarity)
# ==========================================================
class TFIDFService:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def compute_similarity(self, job_description: str, resume_texts: List[str]):
        docs = [job_description] + resume_texts
        tfidf_matrix = self.vectorizer.fit_transform(docs)
        job_vec = tfidf_matrix[0:1]
        resume_vecs = tfidf_matrix[1:]

        sims = cosine_similarity(job_vec, resume_vecs)[0] * 100
        return sims.tolist()

# ==========================================================
# EMAIL SERVICE
# ==========================================================
class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.sender_email = os.getenv("SENDER_EMAIL", "hrteam@gmail.com")

    def send_candidate_email(self, to_address, candidate_name, job_description, score):
        subject = "Congratulations! You Have Been Shortlisted"

        body = f"""
Dear {candidate_name},

You have been shortlisted for the next steps in the selection process.

Regards,
HR Team
NAV Tech.
"""

        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = to_address
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.sender_email, to_address, msg.as_string())
            return True
        except Exception as e:
            raise Exception(f"Email sending failed: {e}")

# ==========================================================
# SCORING SERVICE (TF-IDF + Skills)
# ==========================================================
class ScoringService:
    def __init__(self):
        self.tfidf = TFIDFService()
        #self.semantic_weight = 0.7
        self.skill_weight = 0.3

    def score(self, job_description, resume_items):
        resume_texts = [r["text"] for r in resume_items]
        semantic_scores = self.tfidf.compute_similarity(job_description, resume_texts)

        results = []
        for i, item in enumerate(resume_items):
            jd_skills = extract_skills_from_text(job_description)
            skill_score = len(set(jd_skills).intersection(item["skills"])) / max(len(jd_skills), 1) * 100
            semantic_score = semantic_scores[i]

            final = (
              #  self.semantic_weight * semantic_score +
                self.skill_weight * skill_score
            )

            results.append({
                "name": item["name"],
                "email": item["email"],
                "score": round(final, 2),
                #"semantic_score": round(semantic_score, 2),
                #"skill_score": round(skill_score, 2),
                "skills": item["skills"]
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

scoring_service = ScoringService()
email_service = EmailService()

# ==========================================================
# PROCESS RESUMES
# ==========================================================
def process_resumes(job_description: str, uploaded_files: List):
    resume_items = []
    temp_paths = []

    try:
        for f in uploaded_files:
            if not allowed_file(f.name):
                continue

            tmp = secure_temp_save(f)
            temp_paths.append(tmp)

            text = parse_resume(tmp)
            skills = extract_skills_from_text(text)
            emails = extract_emails_from_text(text)

            resume_items.append({
                "name": os.path.splitext(f.name)[0],
                "text": text,
                "skills": skills,
                "email": emails[0] if emails else None
            })

        return scoring_service.score(job_description, resume_items)

    finally:
        for p in temp_paths:
            try: os.remove(p)
            except: pass

# ==========================================================
# STREAMLIT UI (WITH SESSION STATE)
# ==========================================================
st.set_page_config(page_title="Resume Ranking - TF-IDF", layout="wide")
st.title("Resume Ranking (TF-IDF Version)")

if "results" not in st.session_state:
    st.session_state.results = None

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

if "top_n" not in st.session_state:
    st.session_state.top_n = 3

job_description = st.text_area("Job Description", height=200)
uploaded_files = st.file_uploader("Upload resumes", accept_multiple_files=True)

top_n = st.number_input("Top N", min_value=1, max_value=50, value=3)

# ----------------------------------------------------
# RANK BUTTON
# ----------------------------------------------------
if st.button("Rank Resumes"):
    if not job_description.strip():
        st.error("Enter job description")
        st.stop()
    if not uploaded_files:
        st.error("Upload resumes")
        st.stop()

    with st.spinner("Ranking…"):
        results = process_resumes(job_description, uploaded_files)

    st.session_state.results = results
    st.session_state.job_description = job_description
    st.session_state.top_n = top_n

    st.success("Ranking complete!")

# ----------------------------------------------------
# SHOW RESULTS
# ----------------------------------------------------
if st.session_state.results:
    results = st.session_state.results
    job_description = st.session_state.job_description
    top_n = st.session_state.top_n

    st.write("### 🏆 Top Candidates")

    for i, r in enumerate(results[:int(top_n)], start=1):
        st.markdown(f"**{i}. {r['name']}** — Score: `{r['score']}%` — Email: `{r['email']}`")

    df = pd.DataFrame(results)
    df["skills"] = df["skills"].apply(lambda x: ", ".join(x))
    st.dataframe(df)

    # ----------------------------------------------------
    # SEND EMAIL BUTTON
    # ----------------------------------------------------
    if st.button(f"📧 Send Email to Top {int(top_n)} Candidates"):
        st.write("### Sending Emails…")
        logs = []

        for r in results[:int(top_n)]:
            email = r["email"]
            if not email:
                logs.append({"email": None, "status": "no email"})
                continue

            try:
                email_service.send_candidate_email(email, r["name"], job_description, r["score"])
                logs.append({"email": email, "status": "sent"})
            except Exception as e:
                logs.append({"email": email, "status": f"failed: {e}"})


        st.json(logs)
        st.success("Done!")
