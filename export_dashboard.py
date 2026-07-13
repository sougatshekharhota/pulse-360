"""Export a browser-ready dashboard.json from the pipeline outputs.

Reuses the exact same transform layer the Streamlit app uses, so the static
GitHub Pages dashboard shows identical numbers — just precomputed.
"""
import json
from pathlib import Path

from src import db, transform
from src.config import DB_PATH, entities, load_config

OUT = Path(__file__).resolve().parent / "docs" / "dashboard.json"


def main() -> None:
    cfg = load_config()
    ents = [e["name"] for e in entities(cfg)]
    brand = cfg["brand"]["name"]

    con = db.connect(DB_PATH)
    series = db.load_series(con)
    mentions = db.load_mentions(con)
    updated = db.last_updated(con)

    if not series.empty:
        series = series[series["entity"].isin(ents)]
    if not mentions.empty:
        mentions = mentions[mentions["entity"].isin(ents)]

    scores = transform.health_scores(series, mentions, cfg)
    overall, overall_grade = transform.brand_overall(scores, brand)

    # ordered source keys present
    order = list(transform.SOURCE_LABELS)
    sources = sorted(series["source"].unique(), key=lambda s: order.index(s) if s in order else 99)

    # series[source][entity] = [[date, value], ...]
    series_out: dict = {}
    for src in sources:
        series_out[src] = {}
        sub = series[series["source"] == src]
        for ent in ents:
            g = sub[sub["entity"] == ent].sort_values("date")
            if not g.empty:
                series_out[src][ent] = [[d.strftime("%Y-%m-%d"), round(float(v), 2)]
                                        for d, v in zip(g["date"], g["value"])]

    sov = transform.share_of_voice(series)
    mom = transform.weekly_momentum(series)
    sov_out = {src: [{"entity": r.entity, "sov": round(float(r.sov), 4)}
                     for r in sov[sov["source"] == src].itertuples()] for src in sources}
    mom_out = {src: [{"entity": r.entity, "wow": round(float(r.wow_change), 4)}
                     for r in mom[mom["source"] == src].itertuples()] for src in sources}

    kpis = [{"channel": r.channel, "score": int(r.score), "grade": r.grade,
             "wow": round(float(r.wow_change), 4)}
            for r in scores[scores["entity"] == brand].sort_values("channel").itertuples()]

    scores_out = [{"channel": r.channel, "entity": r.entity, "score": int(r.score), "grade": r.grade}
                  for r in scores.itertuples()]

    mentions_out = []
    if not mentions.empty:
        m = mentions.sort_values("created", ascending=False).head(60)
        for r in m.itertuples():
            mentions_out.append({
                "when": r.created.strftime("%d %b %Y"),
                "brand": r.entity,
                "title": r.title,
                "score": int(r.score) if r.score == r.score else 0,
                "sentiment": round(float(r.sentiment), 3),
            })

    data = {
        "brand": brand,
        "entities": ents,
        "updated": updated,
        "overall": {"score": overall, "grade": overall_grade},
        "sources": sources,
        "sourceLabels": transform.SOURCE_LABELS,
        "kpis": kpis,
        "scores": scores_out,
        "series": series_out,
        "sov": sov_out,
        "momentum": mom_out,
        "mentions": mentions_out,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
