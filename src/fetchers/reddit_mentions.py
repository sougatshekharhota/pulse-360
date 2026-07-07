"""Reddit brand mentions via the public JSON search endpoint (no API key).

Each mention is stored with a VADER sentiment score; the daily mention count
also becomes a time series for the community channel.
"""
from datetime import datetime, timezone

import requests

SEARCH = "https://www.reddit.com/search.json"
# Reddit's public JSON endpoint rejects generic/script user agents with 403;
# a descriptive browser-style UA is required.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 "
        "PULSE-360/1.0 (marketing-analytics portfolio project)"
    ),
    "Accept": "application/json",
}


def fetch(cfg: dict, analyzer) -> tuple[list[tuple], list[tuple]]:
    """Returns (mention_rows, series_rows)."""
    reddit_cfg = cfg.get("reddit", {})
    limit = int(reddit_cfg.get("results_per_brand", 100))
    window = reddit_cfg.get("time_window", "month")

    from ..config import entities
    mention_rows: list[tuple] = []
    counts: dict[tuple[str, str], int] = {}

    for ent in entities(cfg):
        name = ent["name"]
        try:
            resp = requests.get(
                SEARCH,
                params={"q": name, "sort": "new", "limit": limit, "t": window},
                headers=HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[reddit] {name}: {exc}")
            continue
        for child in resp.json().get("data", {}).get("children", []):
            d = child.get("data", {})
            created = datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc)
            title = (d.get("title") or "")[:300]
            sentiment = analyzer.polarity_scores(title)["compound"] if title else 0.0
            mention_rows.append((
                f"reddit:{d.get('id')}",
                "reddit",
                name,
                created.isoformat(),
                title,
                "https://www.reddit.com" + (d.get("permalink") or ""),
                float(d.get("score", 0)),
                sentiment,
            ))
            day = created.strftime("%Y-%m-%d")
            counts[(name, day)] = counts.get((name, day), 0) + 1

    series_rows = [("reddit", name, day, float(n)) for (name, day), n in counts.items()]
    return mention_rows, series_rows
