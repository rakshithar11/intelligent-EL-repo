import os, re, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pymupdf as fitz
import docx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from database import db

RAW = "documents/raw"
INDEX = "database/index.faiss"
CHUNK_WORDS, OVERLAP = 500, 50

DOMAINS = {
    "AI / Machine Learning": "machine learning deep learning neural network model training dataset prediction algorithm",
    "Healthcare / Biomedical": "medical patient cancer clinical diagnosis detection radiology health screening",
    "Energy & Sustainability": "energy efficiency value stream mapping sustainability green manufacturing emissions environmental performance",
    "Manufacturing & Production": "CNC machining manufacturing process production tooling ergonomics assembly workstation",
    "Automotive & Thermal": "engine combustion hydrogen fuel storage vehicle thermal heat transfer",
    "Logistics & Operations": "logistics supply chain scheduling project time cost trade-off resilience optimization",
    "Software & Web Platforms": "web application platform software users mobile app database interface exchange",
    "Aerospace": "aircraft aerodynamics flight wing propulsion structures",
    "Electronics & IoT": "sensor IoT embedded circuit microcontroller wireless automation",
    "Materials": "material alloy composite coating strength microstructure corrosion",
}

_model = None
_dvec = None
_index = None


def model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def domain_vectors():
    global _dvec
    if _dvec is None:
        _dvec = model().encode(list(DOMAINS.values()), normalize_embeddings=True)
    return _dvec


def get_index():
    global _index
    if _index is None:
        if os.path.exists(INDEX):
            _index = faiss.read_index(INDEX)
        else:
            _index = faiss.IndexIDMap(faiss.IndexFlatIP(384))
    return _index


def save_index():
    faiss.write_index(get_index(), INDEX)


def read_text(path):
    if path.lower().endswith(".pdf"):
        d = fitz.open(path)
        t = " ".join(p.get_text() for p in d)
        d.close()
    else:
        t = " ".join(p.text for p in docx.Document(path).paragraphs)
    return re.sub(r"\s+", " ", t).strip()


def make_chunks(text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), step)
            if words[i:i + CHUNK_WORDS]]


def classify(text):
    sample = text[:6000]
    parts = [sample[i:i + 800] for i in range(0, len(sample), 800)] or [""]
    v = model().encode(parts, normalize_embeddings=True).mean(axis=0)
    return list(DOMAINS)[int(np.argmax(domain_vectors() @ v))]


def guess_year(text):
    ys = [int(y) for y in re.findall(r"\b(20[0-2]\d)\b", text[:3000])]
    ys = [y for y in ys if y <= 2026]
    return str(Counter(ys).most_common(1)[0][0]) if ys else ""


def add_document(filename, file_bytes=None, uploaded_by="", team="", mentor="",
                 year="", domain_override=""):
    """Returns (ok, message). If file_bytes is None, file must already be in RAW."""
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, filename)

    if db.filename_exists(filename):
        return False, f"'{filename}' already exists."
    if file_bytes is not None:
        with open(path, "wb") as f:
            f.write(file_bytes)

    text = read_text(path)
    chunks = make_chunks(text)
    if not chunks:
        if file_bytes is not None:
            os.remove(path)
        return False, f"No readable text in '{filename}' (scanned file?)."

    pid = db.add_project(
        title=os.path.splitext(filename)[0],
        domain_auto=classify(text),
        year=year or guess_year(text),
        filename=filename, team=team, mentor=mentor, uploaded_by=uploaded_by)
    if domain_override:
        db.update_project(pid, domain_override=domain_override)

    ids = db.add_chunks(pid, chunks)
    vecs = model().encode(chunks, batch_size=32, normalize_embeddings=True)
    get_index().add_with_ids(np.array(vecs, dtype="float32"),
                             np.array(ids, dtype="int64"))
    save_index()
    return True, f"Added '{filename}' ({len(chunks)} chunks)."


def remove_document(pid, delete_file=True):
    p = db.get_project(pid)
    if not p:
        return False
    ids = db.delete_project(pid)
    if ids:
        get_index().remove_ids(np.array(ids, dtype="int64"))
        save_index()
    if delete_file:
        fp = os.path.join(RAW, p["filename"])
        if os.path.exists(fp):
            os.remove(fp)
    return True


def load_existing_folder():
    """Adds every file in documents/raw that is not yet in the database."""
    for f in sorted(os.listdir(RAW)):
        if f.lower().endswith((".pdf", ".docx")) and not db.filename_exists(f):
            print("Processing:", f)
            ok, msg = add_document(f)
            print("  ", msg)


if __name__ == "__main__":
    db.init_db()
    load_existing_folder()
    print("Total projects in database:", len(db.list_projects()))