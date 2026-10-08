from flask import Flask, render_template, request, send_file
import os
import re
import sqlite3
from io import BytesIO

from pypdf import PdfReader
from docx import Document

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_FILE = "screening.db"


# =========================================================
# DATABASE
# =========================================================

def init_db():

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            education TEXT,
            job_role TEXT,
            qualification TEXT,
            skill_score INTEGER,
            ai_score INTEGER,
            score INTEGER,
            status TEXT,
            matched_skills TEXT,
            missing_skills TEXT,
            job_description TEXT,
            resume_text TEXT,
            resume_filename TEXT
        )
    """)

    existing_columns = [
        row[1]
        for row in cursor.execute(
            "PRAGMA table_info(candidates)"
        ).fetchall()
    ]

    new_columns = {
        "qualification": "TEXT",
        "skill_score": "INTEGER",
        "ai_score": "INTEGER",
        "matched_skills": "TEXT",
        "missing_skills": "TEXT",
        "job_description": "TEXT",
        "resume_text": "TEXT",
        "resume_filename": "TEXT"
    }

    for column, data_type in new_columns.items():

        if column not in existing_columns:

            cursor.execute(
                f"ALTER TABLE candidates ADD COLUMN {column} {data_type}"
            )

    conn.commit()
    conn.close()


init_db()


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(file_path):

    text = ""

    reader = PdfReader(file_path)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# DOCX TEXT EXTRACTION
# =========================================================

def extract_docx_text(file_path):

    document = Document(file_path)

    text = ""

    for paragraph in document.paragraphs:

        text += paragraph.text + "\n"

    return text


# =========================================================
# CANDIDATE INFORMATION EXTRACTION
# =========================================================

def extract_candidate_info(text):

    # EMAIL

    email_match = re.search(
        r'[\w\.-]+@[\w\.-]+\.\w+',
        text
    )

    if email_match:
        email = email_match.group(0)
    else:
        email = "Not Found"


    # PHONE

    phone_match = re.search(
        r'(\+91[\s-]?)?[6-9]\d{9}',
        text
    )

    if phone_match:
        phone = phone_match.group(0)
    else:
        phone = "Not Found"


    # NAME

    name = "Not Found"

    for line in text.splitlines():

        line = line.strip()

        if line.lower().startswith("name:"):

            name = line.split(":", 1)[1].strip()

            break


    # EDUCATION

    education = "Not Found"

    education_keywords = [
        "b.sc",
        "bsc",
        "bachelor",
        "computer science",
        "m.sc",
        "msc",
        "bca",
        "mca"
    ]

    for line in text.splitlines():

        if any(
            keyword in line.lower()
            for keyword in education_keywords
        ):

            education = line.strip()

            break


    return name, email, phone, education


# =========================================================
# AI SIMILARITY
# =========================================================

def calculate_ai_score(job_description, resume_text):

    if not job_description.strip():
        return 0

    if not resume_text.strip():
        return 0

    documents = [
        job_description.lower(),
        resume_text.lower()
    ]

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        vectors = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        score = round(similarity * 100)

        return score

    except Exception:

        return 0


# =========================================================
# HOME / DASHBOARD
# =========================================================

@app.route("/")
def home():

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()


    total_candidates = cursor.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]


    average_score = cursor.execute(
        "SELECT AVG(score) FROM candidates"
    ).fetchone()[0]


    if average_score is None:
        average_score = 0
    else:
        average_score = round(average_score)


    accepted_candidates = cursor.execute(
        """
        SELECT COUNT(*)
        FROM candidates
        WHERE status = 'ACCEPTED'
        """
    ).fetchone()[0]


    review_count = cursor.execute(
        """
        SELECT COUNT(*)
        FROM candidates
        WHERE status = 'REVIEW'
        """
    ).fetchone()[0]


    rejected_count = cursor.execute(
        """
        SELECT COUNT(*)
        FROM candidates
        WHERE status = 'REJECTED'
        """
    ).fetchone()[0]


    conn.close()


    return render_template(
        "index.html",
        candidates_screened=total_candidates,
        resumes_analyzed=total_candidates,
        average_score=average_score,
        accepted_candidates=accepted_candidates,
        review_count=review_count,
        rejected_count=rejected_count
    )


# =========================================================
# UPLOAD / SCREEN MULTIPLE RESUMES
# =========================================================

@app.route("/upload", methods=["POST"])
def upload():

    job_role = request.form.get(
        "job_role",
        ""
    )


    required_skills = request.form.get(
        "skills",
        ""
    )


    qualification = request.form.get(
        "qualification",
        ""
    )


    job_description = request.form.get(
        "job_description",
        ""
    )


    files = request.files.getlist("resumes")


    if not files:

        return "Please select at least one resume."


    skills = [
        skill.strip().lower()
        for skill in required_skills.split(",")
        if skill.strip()
    ]


    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()


    processed_candidates = []


    for file in files:

        if file.filename == "":
            continue


        if not file.filename.lower().endswith(
            (".pdf", ".docx")
        ):
            continue


        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )


        file.save(file_path)


        try:

            if file.filename.lower().endswith(".pdf"):

                resume_text = extract_pdf_text(
                    file_path
                )

            else:

                resume_text = extract_docx_text(
                    file_path
                )

        except Exception:

            continue


        resume_lower = resume_text.lower()


        name, email, phone, education = extract_candidate_info(
            resume_text
        )


        # SKILL MATCHING

        matched_skills = []
        missing_skills = []


        for skill in skills:

            if skill in resume_lower:

                matched_skills.append(skill)

            else:

                missing_skills.append(skill)


        if len(skills) > 0:

            skill_score = round(
                (len(matched_skills) / len(skills)) * 100
            )

        else:

            skill_score = 0


        # AI SCORE

        ai_score = calculate_ai_score(
            job_description,
            resume_text
        )


        # FINAL SCORE

        if job_description.strip():

            final_score = round(
                (skill_score * 0.5) +
                (ai_score * 0.5)
            )

        else:

            final_score = skill_score


        # STATUS

        if final_score >= 70:

            status = "ACCEPTED"

        elif final_score >= 50:

            status = "REVIEW"

        else:

            status = "REJECTED"


        # SAVE TO DATABASE

        cursor.execute("""
            INSERT INTO candidates
            (
                name,
                email,
                phone,
                education,
                job_role,
                qualification,
                skill_score,
                ai_score,
                score,
                status,
                matched_skills,
                missing_skills,
                job_description,
                resume_text,
                resume_filename
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            name,
            email,
            phone,
            education,
            job_role,
            qualification,
            skill_score,
            ai_score,
            final_score,
            status,
            ", ".join(matched_skills),
            ", ".join(missing_skills),
            job_description,
            resume_text,
            file.filename

        ))


        candidate_id = cursor.lastrowid


        processed_candidates.append({

            "id": candidate_id,

            "name": name,

            "score": final_score,

            "status": status

        })


    conn.commit()
    conn.close()


    return render_template(
        "multiple_result.html",
        candidates=processed_candidates,
        job_role=job_role
    )


# =========================================================
# CANDIDATES DATABASE
# =========================================================

@app.route("/candidates")
def candidates():

    conn = sqlite3.connect(DB_FILE)

    conn.row_factory = sqlite3.Row


    candidates = conn.execute("""
        SELECT *
        FROM candidates
        ORDER BY id DESC
    """).fetchall()


    conn.close()


    return render_template(
        "candidates.html",
        candidates=candidates
    )


# =========================================================
# CANDIDATE DETAILS
# =========================================================

@app.route("/candidate/<int:candidate_id>")
def candidate_detail(candidate_id):

    conn = sqlite3.connect(DB_FILE)

    conn.row_factory = sqlite3.Row


    candidate = conn.execute(
        """
        SELECT *
        FROM candidates
        WHERE id = ?
        """,
        (candidate_id,)
    ).fetchone()


    conn.close()


    if candidate is None:

        return "Candidate not found."


    return render_template(
        "candidate_detail.html",
        candidate=candidate
    )


# =========================================================
# DOWNLOAD PDF REPORT
# =========================================================

@app.route("/download_report/<int:candidate_id>")
def download_report(candidate_id):

    conn = sqlite3.connect(DB_FILE)

    conn.row_factory = sqlite3.Row


    candidate = conn.execute(
        """
        SELECT *
        FROM candidates
        WHERE id = ?
        """,
        (candidate_id,)
    ).fetchone()


    conn.close()


    if candidate is None:

        return "Candidate not found."


    # CREATE PDF IN MEMORY

    buffer = BytesIO()


    pdf = canvas.Canvas(
        buffer,
        pagesize=A4
    )


    width, height = A4

    y = height - 50


    # TITLE

    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawString(
        50,
        y,
        "HireSense AI"
    )


    y -= 30


    pdf.setFont(
        "Helvetica-Bold",
        15
    )

    pdf.drawString(
        50,
        y,
        "Candidate Screening Report"
    )


    y -= 35


    pdf.setFont(
        "Helvetica",
        11
    )


    lines = [

        f"Name: {candidate['name']}",

        f"Email: {candidate['email']}",

        f"Phone: {candidate['phone']}",

        f"Education: {candidate['education']}",

        f"Job Role: {candidate['job_role']}",

        f"Qualification: {candidate['qualification']}",

        f"Skill Match Score: {candidate['skill_score']}%",

        f"AI Similarity Score: {candidate['ai_score']}%",

        f"Final Screening Score: {candidate['score']}%",

        f"Status: {candidate['status']}",

        f"Matched Skills: {candidate['matched_skills']}",

        f"Missing Skills: {candidate['missing_skills']}"

    ]


    for line in lines:

        if y < 50:

            pdf.showPage()

            pdf.setFont(
                "Helvetica",
                11
            )

            y = height - 50


        pdf.drawString(
            50,
            y,
            str(line)[:110]
        )


        y -= 20


    pdf.setFont(
        "Helvetica-Oblique",
        9
    )


    pdf.drawString(
        50,
        30,
        "Generated by HireSense AI | Developed by Prasad"
    )


    pdf.save()


    buffer.seek(0)


    safe_name = re.sub(
        r'[^A-Za-z0-9_-]',
        '_',
        str(candidate["name"])
    )


    return send_file(

        buffer,

        as_attachment=True,

        download_name=
        f"{safe_name}_HireSense_Report.pdf",

        mimetype="application/pdf"

    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )