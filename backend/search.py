import os, re, sys, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
from database import db
from ml import ingest

MIN_SEM = 0.25  # raise to be stricter, lower to see more results
STOP = {"the", "and", "for", "using", "with", "from", "that", "this", "are",
        "how", "what", "into", "about", "based", "study", "its", "can"}


def dom_ok(p, domain):
    if not domain or domain == "All":
        return True
    return domain.lower() in p["domain"].lower()


def _package(pid, score, snippet):
    p = db.get_project(pid)
    if not p:
        return None
    p["score"] = round(float(score), 4)
    p["snippet"] = snippet[:300] + "..."
    return p


def semantic_search(query, k=5, domain=None):
    index = ingest.get_index()
    if index.ntotal == 0:
        return []
    qv = ingest.model().encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.array(qv, dtype="float32"),
                               min(index.ntotal, 200))
    pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0])
             if i != -1 and s >= MIN_SEM]
    chunks = {c["chunk_id"]: c
              for c in db.get_chunks_by_ids([i for i, _ in pairs])}
    best = {}
    for cid, s in pairs:
        c = chunks.get(cid)
        if c and c["project_id"] not in best:
            best[c["project_id"]] = (s, c["text"])
    out = []
    for pid, (s, txt) in best.items():
        p = _package(pid, s, txt)
        if p and dom_ok(p, domain):
            out.append(p)
    return out[:k]


def keyword_search(query, k=5, domain=None):
    words = [w for w in re.findall(r"\w+", query.lower())
             if len(w) > 2 and w not in STOP]
    if not words:
        return []
    pats = {w: re.compile(r"\b" + re.escape(w) + r"(s|es)?\b") for w in words}

    data = {}
    with db.conn() as c:
        for r in c.execute("SELECT project_id, text FROM chunks"):
            t = r["text"].lower()
            d = data.setdefault(r["project_id"],
                                {"words": set(), "score": 0, "best": (0, r["text"])})
            s = 0
            for w, pat in pats.items():
                n = len(pat.findall(t))
                if n:
                    d["words"].add(w)
                    s += min(n, 3)
            d["score"] += s
            if s > d["best"][0]:
                d["best"] = (s, r["text"])

    need = math.ceil(0.6 * len(words))
    ranked = []
    for p in db.list_projects():
        d = data.get(p["id"], {"words": set(), "score": 0, "best": (0, p["title"])})
        meta = (p["title"] + " " + p["domain"]).lower()
        for w, pat in pats.items():
            if pat.search(meta):
                d["words"].add(w)
                d["score"] += 5
        if len(d["words"]) >= need and dom_ok(p, domain):
            ranked.append((d["score"], p["id"], d["best"][1]))
    ranked.sort(reverse=True)

    out = []
    for s, pid, txt in ranked[:k]:
        p = _package(pid, s, txt)
        if p:
            out.append(p)
    return out


def search(query, k=5, domain=None, mode="hybrid"):
    if mode == "semantic":
        return semantic_search(query, k, domain)
    if mode == "keyword":
        return keyword_search(query, k, domain)
    sem = semantic_search(query, 20, domain)
    kw = keyword_search(query, 20, domain)
    fused, info = {}, {}
    for lst in (sem, kw):
        for rank, p in enumerate(lst):
            fused[p["id"]] = fused.get(p["id"], 0) + 1 / (60 + rank)
            info.setdefault(p["id"], p)
    top = sorted(fused, key=lambda i: -fused[i])[:k]
    results = []
    for i in top:
        p = info[i]
        p["score"] = round(fused[i], 4)
        results.append(p)
    return results


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "hydrogen engine storage"
    for r in search(q):
        print(f"{r['score']:.3f} | {r['domain']} | {r['title'][:60]}")