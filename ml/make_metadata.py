import os, re, csv
from collections import Counter
import fitz  # pymupdf
import docx
import numpy as np
from sentence_transformers import SentenceTransformer

RAW = "documents/raw"
OUT = "documents/metadata.csv"
COLS = ["id", "title", "auto_domain", "domain_override",
        "year", "team", "mentor", "filename"]

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

def read_text(path):
    if path.lower().endswith(".pdf"):
        d = fitz.open(path)
        t = " ".join(d[i].get_text() for i in range(min(6, len(d))))
        d.close()
    else:
        t = " ".join(p.text for p in docx.Document(path).paragraphs)
    return re.sub(r"\s+", " ", t)[:6000]

model = SentenceTransformer("all-MiniLM-L6-v2")
names = list(DOMAINS)
dvec = model.encode(list(DOMAINS.values()), normalize_embeddings=True)

def classify(text):
    chunks = [text[i:i + 800] for i in range(0, len(text), 800)] or [""]
    v = model.encode(chunks, normalize_embeddings=True).mean(axis=0)
    return names[int(np.argmax(dvec @ v))]

def guess_year(text):
    ys = [int(y) for y in re.findall(r"\b(20[0-2]\d)\b", text[:3000])]
    ys = [y for y in ys if y <= 2026]
    return Counter(ys).most_common(1)[0][0] if ys else ""

old = {}
if os.path.exists(OUT):
    with open(OUT, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            old[r["filename"]] = r

rows = []
for f in sorted(os.listdir(RAW)):
    if not f.lower().endswith((".pdf", ".docx")):
        continue
    if f in old:
        rows.append(old[f])
        continue
    print("Reading:", f)
    text = read_text(os.path.join(RAW, f))
    rows.append({
        "title": os.path.splitext(f)[0],
        "auto_domain": classify(text),
        "domain_override": "",
        "year": guess_year(text),
        "team": "", "mentor": "",
        "filename": f,
    })

for i, r in enumerate(rows, 1):
    r["id"] = i

with open(OUT, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore", restval="")
    w.writeheader()
    w.writerows(rows)

print("Done:", len(rows), "files")