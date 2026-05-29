import sqlite3
import uuid
from datetime import datetime
from config import DB_PATH


def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS recommendation_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id        TEXT,
                timestamp         DATETIME DEFAULT CURRENT_TIMESTAMP,
                tab               TEXT,
                method            TEXT,
                input_user_id     INTEGER,
                seed_movie        TEXT,
                recommended_title TEXT,
                predicted_score   REAL,
                language_filter   TEXT,
                mood_filter       TEXT
            );
            CREATE TABLE IF NOT EXISTS session_log (
                session_id  TEXT PRIMARY KEY,
                timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
                tab_used    TEXT,
                method_used TEXT
            );
        """)


def new_session_id() -> str:
    return str(uuid.uuid4())[:8]


def log_recommendations(session_id: str, tab: str, method: str,
                        recommendations: list[tuple],
                        input_user_id: int = None, seed_movie: str = None,
                        language_filter: str = None, mood_filter: str = None):
    with _conn() as con:
        con.execute(
            "INSERT OR IGNORE INTO session_log (session_id, tab_used, method_used) VALUES (?,?,?)",
            (session_id, tab, method),
        )
        con.executemany(
            """INSERT INTO recommendation_log
               (session_id, tab, method, input_user_id, seed_movie,
                recommended_title, predicted_score, language_filter, mood_filter)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [
                (session_id, tab, method, input_user_id, seed_movie,
                 title, score, language_filter, mood_filter)
                for title, score in recommendations
            ],
        )


def get_analytics() -> dict:
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) FROM recommendation_log").fetchone()[0]

        top_movies = con.execute("""
            SELECT recommended_title, COUNT(*) as cnt
            FROM recommendation_log
            GROUP BY recommended_title
            ORDER BY cnt DESC LIMIT 10
        """).fetchall()

        daily = con.execute("""
            SELECT DATE(timestamp) as day, COUNT(*) as cnt
            FROM recommendation_log
            GROUP BY day ORDER BY day DESC LIMIT 14
        """).fetchall()

        seed_movies = con.execute("""
            SELECT seed_movie, COUNT(*) as cnt
            FROM recommendation_log
            WHERE seed_movie IS NOT NULL
            GROUP BY seed_movie ORDER BY cnt DESC LIMIT 10
        """).fetchall()

    return {
        "total":       total,
        "top_movies":  [dict(r) for r in top_movies],
        "daily":       [dict(r) for r in daily],
        "seed_movies": [dict(r) for r in seed_movies],
    }
