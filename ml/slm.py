import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import ollama
from database import db
from ml import ingest

MODEL = "llama3.2:1b"   # faster than 3b
MIN_SCORE = 0.20
CHUNK_CHARS = 800       # less text sent to the model

SYSTEM = (
    "You are the research assistant of the RVCE Experiential Learning (EL) "
    "repository. Answer ONLY from the repository excerpts provided. "
    "Each excerpt is numbered like [1]. Cite the project title when you use it. "
    "If the excerpts do not contain the answer, say so clearly. Never invent "
    "projects, results or numbers. "
    "When asked for similar projects, list the matching project titles and say "
    "in one line what each one did. "
    "When asked for research gaps, compare the projects and point out what is "
    "missing, untested or could be extended, based only on the excerpts. "
    "Keep answers short and structured."
)


def retrieve(question, k=3, per_project=2):
    index = ingest.get_index()
    if index.ntotal == 0:
        return []
    qv = ingest.model().encode([question], normalize_embeddings=True)
    scores, ids = index.search(np.array(qv, dtype="float32"),
                               min(index.ntotal, 40))
    pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0])
             if i != -1 and s >= MIN_SCORE]
    chunks = {c["chunk_id"]: c
              for c in db.get_chunks_by_ids([i for i, _ in pairs])}
    hits, count = [], {}
    for cid, s in pairs:
        c = chunks.get(cid)
        if not c:
            continue
        pid = c["project_id"]
        if count.get(pid, 0) >= per_project:
            continue
        p = db.get_project(pid)
        if not p:
            continue
        count[pid] = count.get(pid, 0) + 1
        hits.append({"project": p, "text": c["text"], "score": s})
        if len(hits) >= k:
            break
    return hits


def build_context(hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        p = h["project"]
        head = (f"[{i}] Title: {p['title']} | Domain: {p['domain']} | "
                f"Year: {p['year'] or 'n/a'} | Team: {p['team'] or 'n/a'} | "
                f"Mentor: {p['mentor'] or 'n/a'}")
        blocks.append(head + "\n" + h["text"][:CHUNK_CHARS])
    return "\n\n".join(blocks)


def ask(question, history=None, k=3):
    """Returns (token_stream, hits). Use st.write_stream(token_stream)."""
    history = history or []
    search_text = question
    if history:
        last_user = [m["content"] for m in history if m["role"] == "user"]
        if last_user:
            search_text = last_user[-1] + " " + question

    hits = retrieve(search_text, k=k)
    if not hits:
        msg = ("I could not find relevant projects in the repository for that "
               "question. Try different keywords.")
        return iter([msg]), []

    messages = [{"role": "system", "content": SYSTEM}]
    messages += [{"role": m["role"], "content": m["content"]}
                 for m in history[-4:]]
    messages.append({
        "role": "user",
        "content": f"Repository excerpts:\n\n{build_context(hits)}\n\n"
                   f"Question: {question}"
    })

    def stream():
        try:
            for part in ollama.chat(model=MODEL, messages=messages, stream=True,
                                    options={"temperature": 0.2,
                                             "num_ctx": 4096}):
                yield part["message"]["content"]
        except Exception as e:
            yield (f"\n\nThe AI model is not available ({type(e).__name__}). "
                   f"Make sure Ollama is running and the model is installed: "
                   f"ollama pull {MODEL}")

    return stream(), hits


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which projects deal with hydrogen engines?"
    tokens, hits = ask(q)
    for t in tokens:
        print(t, end="", flush=True)
    print("\n\nSources:")
    for h in hits:
        print("-", h["project"]["title"])
