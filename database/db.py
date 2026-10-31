import sqlite3, os
from datetime import datetime

DB_PATH = os.path.join("database", "el.db")


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS projects(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            domain_auto TEXT,
            domain_override TEXT DEFAULT '',
            year TEXT DEFAULT '',
            team TEXT DEFAULT '',
            mentor TEXT DEFAULT '',
            filename TEXT UNIQUE,
            uploaded_by TEXT DEFAULT '',
            uploaded_at TEXT
        );
        CREATE TABLE IF NOT EXISTS chunks(
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            text TEXT
        );
        """)


# domain shown on the site: override if filled, else auto
DOMAIN_SQL = "COALESCE(NULLIF(domain_override,''), domain_auto)"


def add_project(title, domain_auto, year, filename, team="", mentor="", uploaded_by=""):
    with conn() as c:
        cur = c.execute(
            "INSERT INTO projects(title,domain_auto,year,team,mentor,filename,uploaded_by,uploaded_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (title, domain_auto, year, team, mentor, filename, uploaded_by,
             datetime.now().strftime("%Y-%m-%d %H:%M")))
        return cur.lastrowid


def update_project(pid, **fields):
    allowed = {"title", "domain_override", "year", "team", "mentor"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE projects SET {sets} WHERE id=?", (*fields.values(), pid))


def delete_project(pid):
    with conn() as c:
        ids = [r["chunk_id"] for r in c.execute(
            "SELECT chunk_id FROM chunks WHERE project_id=?", (pid,))]
        c.execute("DELETE FROM chunks WHERE project_id=?", (pid,))
        c.execute("DELETE FROM projects WHERE id=?", (pid,))
    return ids  # used later to remove vectors from the search index


def add_chunks(project_id, texts):
    ids = []
    with conn() as c:
        for t in texts:
            cur = c.execute("INSERT INTO chunks(project_id,text) VALUES (?,?)",
                            (project_id, t))
            ids.append(cur.lastrowid)
    return ids  # same ids go into the search index


def list_projects(domain=None, search=None):
    q = f"SELECT *, {DOMAIN_SQL} AS domain FROM projects WHERE 1=1"
    p = []
    if domain and domain != "All":
        q += f" AND {DOMAIN_SQL}=?"
        p.append(domain)
    if search:
        q += " AND (title LIKE ? OR team LIKE ? OR mentor LIKE ?)"
        p += [f"%{search}%"] * 3
    q += " ORDER BY uploaded_at DESC"
    with conn() as c:
        return [dict(r) for r in c.execute(q, p)]


def get_project(pid):
    with conn() as c:
        r = c.execute(f"SELECT *, {DOMAIN_SQL} AS domain FROM projects WHERE id=?",
                      (pid,)).fetchone()
        return dict(r) if r else None


def get_chunks_by_ids(ids):
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    with conn() as c:
        return [dict(r) for r in c.execute(
            f"SELECT chunk_id, project_id, text FROM chunks WHERE chunk_id IN ({marks})", ids)]


def list_domains():
    with conn() as c:
        return [r[0] for r in c.execute(
            f"SELECT DISTINCT {DOMAIN_SQL} FROM projects ORDER BY 1") if r[0]]


def filename_exists(filename):
    with conn() as c:
        return c.execute("SELECT 1 FROM projects WHERE filename=?",
                         (filename,)).fetchone() is not None


if __name__ == "__main__":
    init_db()
    print("Database ready:", DB_PATH)