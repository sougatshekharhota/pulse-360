"""Google Trends interest-over-time via pytrends (unofficial, best-effort).

pytrends is an unofficial client and Google throttles it aggressively; the
fetch runner treats failures here as non-fatal so the pipeline still completes
with the other sources.
"""


def fetch(cfg: dict) -> list[tuple]:
    from pytrends.request import TrendReq  # imported lazily — optional dependency

    from ..config import entities
    names = [e["name"] for e in entities(cfg)][:5]  # Trends compares max 5 terms

    pytrends = TrendReq(hl="en-US", tz=330)  # tz 330 = IST
    pytrends.build_payload(names, timeframe="today 3-m")
    df = pytrends.interest_over_time()
    if df.empty:
        return []

    rows: list[tuple] = []
    for ts, row in df.iterrows():
        iso = ts.strftime("%Y-%m-%d")
        for name in names:
            if name in row:
                rows.append(("trends", name, iso, float(row[name])))
    return rows
