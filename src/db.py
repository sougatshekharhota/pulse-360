"""SQLite storage — one normalized schema for every source."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    source  TEXT NOT NULL,          -- trends | wikipedia | reddit | ga4 | ...
    entity  TEXT NOT NULL,          -- brand or competitor name
    date    TEXT NOT NULL,          -- YYYY-MM-DD
    value   REAL NOT NULL,
    PRIMARY KEY (source, entity, date)
);
CREATE TABLE IF NOT EXISTS mentions (
    id        TEXT PRIMARY KEY,     -- source-native id, prefixed
    source    TEXT NOT NULL,
    entity    TEXT NOT NULL,
    created   TEXT NOT NULL,        -- ISO timestamp
    title     TEXT,
    url       TEXT,
    score     REAL,                 -- source-native engagement (e.g. reddit score)
    sentiment REAL                  -- VADER compound, -1..1
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def upsert_series(con: sqlite3.Connection, rows: list[tuple]) -> int:
    """rows: (source, entity, date, value)"""
    con.executemany(
        "INSERT INTO series (source, entity, date, value) VALUES (?,?,?,?) "
        "ON CONFLICT(source, entity, date) DO UPDATE SET value=excluded.value",
        rows,
    )
    con.commit()
    return len(rows)


def upsert_mentions(con: sqlite3.Connection, rows: list[tuple]) -> int:
    """rows: (id, source, entity, created, title, url, score, sentiment)"""
    con.executemany(
        "INSERT INTO mentions (id, source, entity, created, title, url, score, sentiment) "
        "VALUES (?,?,?,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET score=excluded.score, sentiment=excluded.sentiment",
        rows,
    )
    con.commit()
    return len(rows)


def touch_last_updated(con: sqlite3.Connection) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    con.execute(
        "INSERT INTO meta (key, value) VALUES ('last_updated', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (now,),
    )
    con.commit()


def load_series(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM series", con)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_mentions(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM mentions", con)
    if not df.empty:
        df["created"] = pd.to_datetime(df["created"], format="ISO8601", utc=True)
    return df


def last_updated(con: sqlite3.Connection) -> str:
    row = con.execute("SELECT value FROM meta WHERE key='last_updated'").fetchone()
    return row[0] if row else "never"
