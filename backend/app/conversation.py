"""Conversation history. Stores text only - never camera or microphone recordings.

SQLite by default; MongoDB if MONGO_URI is set.
"""
import sqlite3
import threading
import time
from pathlib import Path

from . import config


class SQLiteStore:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, speaker TEXT, "
                "text TEXT, label TEXT, confidence REAL)"
            )
            self._conn.commit()

    def add(self, speaker: str, text: str, label=None, confidence=None) -> dict:
        ts = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages (ts, speaker, text, label, confidence) VALUES (?,?,?,?,?)",
                (ts, speaker, text, label, confidence),
            )
            self._conn.commit()
        return {"id": cur.lastrowid, "ts": ts, "speaker": speaker, "text": text,
                "label": label, "confidence": confidence}

    def history(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, speaker, text, label, confidence FROM messages "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("id", "ts", "speaker", "text", "label", "confidence")
        return [dict(zip(keys, r, strict=True)) for r in reversed(rows)]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages")
            self._conn.commit()


class MongoStore:
    def __init__(self, uri: str):
        from pymongo import MongoClient

        self._col = MongoClient(uri, serverSelectionTimeoutMS=3000)["signbridge"]["messages"]

    def add(self, speaker: str, text: str, label=None, confidence=None) -> dict:
        doc = {"ts": time.time(), "speaker": speaker, "text": text,
               "label": label, "confidence": confidence}
        res = self._col.insert_one(dict(doc))
        return {"id": str(res.inserted_id), **doc}

    def history(self, limit: int = 50) -> list[dict]:
        docs = list(self._col.find().sort("_id", -1).limit(limit))
        out = []
        for d in reversed(docs):
            d["id"] = str(d.pop("_id"))
            out.append(d)
        return out

    def clear(self) -> None:
        self._col.delete_many({})


def get_store():
    if config.MONGO_URI:
        return MongoStore(config.MONGO_URI)
    return SQLiteStore(config.DB_PATH)
