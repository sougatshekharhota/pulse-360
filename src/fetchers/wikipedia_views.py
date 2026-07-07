"""Wikipedia pageviews — a free, keyless proxy for brand research interest.

Uses the official Wikimedia REST API:
https://wikimedia.org/api/rest_v1/metrics/pageviews/

Article titles must be percent-encoded (parentheses and slashes included),
and one missing article must not sink the other entities — each article is
fetched independently.
"""
from datetime import date, timedelta
from urllib.parse import quote

import requests

API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}"
)
HEADERS = {"User-Agent": "PULSE-360 portfolio project (github.com/sougatshekhar97-cpu/pulse-360)"}


def fetch(cfg: dict) -> list[tuple]:
    lookback = int(cfg.get("lookback_days", 90))
    end = date.today() - timedelta(days=1)          # pageviews lag ~1 day
    start = end - timedelta(days=lookback)
    rows: list[tuple] = []

    from ..config import entities
    for ent in entities(cfg):
        article = ent.get("wikipedia_article")
        if not article:
            continue
        url = API.format(
            article=quote(article, safe=""),
            start=start.strftime("%Y%m%d00"),
            end=end.strftime("%Y%m%d00"),
        )
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[wikipedia] {ent['name']} ({article}): {exc}")
            continue
        for item in resp.json().get("items", []):
            ts = item["timestamp"]                   # YYYYMMDD00
            iso = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"
            rows.append(("wikipedia", ent["name"], iso, float(item["views"])))
    return rows
