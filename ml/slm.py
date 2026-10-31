import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import ollama
from database import db
from ml import ingest

MODEL = "llama3.2:3b"
MIN_SCORE = 0.30        # only strong matches reach the model
CHUNK_CHARS = 900

SYSTEM = (
    "You are the research assistant of the RVCE Experiential Learning (EL) "
    "repository. Use ONLY the numbered excerpts provided. Ignore any excerpt "
    "that is not clearly about the question. Always refer to a project by its "
    "full title, never just by its number. "
    "Format: for each relevant project write one line: "
    "**Title** (Domain, Year): one sentence on what it did or found. "
    "If the question asks for research gaps, add a short 'Research gaps' list "
    "of 2-3 points based only on the excerpts. "
    "If the user gives only a topic, list the relevant projects in the same "
    "format. "
    "If no excerpt is relevant, reply exactly: No relevant projects found in "
    "the repository. "
    "Never invent projects, results or numbers."
)


def retrieve(question, k=3, per_project=1):
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
        msg = ("No relevant projects found in the repository. "
               "Try different keywords.")
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
                                    options={"temperature": 0.1,
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
