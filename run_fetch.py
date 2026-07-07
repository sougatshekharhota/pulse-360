"""PULSE-360 fetch runner.

Usage:
  python run_fetch.py            # fetch live data from every available source
  python run_fetch.py --demo     # seed deterministic demo data (no network)
  python run_fetch.py --sources wikipedia,reddit

Each source is isolated: one failing (rate limits, outages) never blocks the
rest. Exit code is 0 as long as at least one source succeeded.
"""
import argparse
import sys
import traceback

from src import db
from src.config import load_config

ALL_SOURCES = ["wikipedia", "news", "trends", "reddit", "ga4"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch marketing signals into data/pulse.db")
    parser.add_argument("--demo", action="store_true", help="seed demo data instead of fetching")
    parser.add_argument("--sources", default=",".join(ALL_SOURCES),
                        help=f"comma list from: {','.join(ALL_SOURCES)}")
    args = parser.parse_args()

    cfg = load_config()
    con = db.connect()
    ok: list[str] = []
    failed: list[str] = []

    if args.demo:
        from src.demo import generate
        series_rows, mention_rows = generate(cfg)
        db.upsert_series(con, series_rows)
        db.upsert_mentions(con, mention_rows)
        db.touch_last_updated(con)
        print(f"[demo] seeded {len(series_rows)} series points, {len(mention_rows)} mentions")
        return 0

    wanted = [s.strip() for s in args.sources.split(",") if s.strip()]

    if "wikipedia" in wanted:
        try:
            from src.fetchers import wikipedia_views
            n = db.upsert_series(con, wikipedia_views.fetch(cfg))
            ok.append(f"wikipedia ({n} points)")
        except Exception:
            failed.append("wikipedia")
            traceback.print_exc()

    if "news" in wanted:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            from src.fetchers import gdelt_news
            mentions, series = gdelt_news.fetch(cfg, SentimentIntensityAnalyzer())
            if mentions:
                db.upsert_mentions(con, mentions)
                db.upsert_series(con, series)
                ok.append(f"news ({len(mentions)} mentions)")
            else:
                failed.append("news (0 articles returned)")
        except Exception:
            failed.append("news")
            traceback.print_exc()

    if "reddit" in wanted:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            from src.fetchers import reddit_mentions
            mentions, series = reddit_mentions.fetch(cfg, SentimentIntensityAnalyzer())
            if mentions:
                db.upsert_mentions(con, mentions)
                db.upsert_series(con, series)
                ok.append(f"reddit ({len(mentions)} mentions)")
            else:
                failed.append("reddit (blocked or empty — Reddit requires OAuth for most scripted access)")
        except Exception:
            failed.append("reddit")
            traceback.print_exc()

    if "trends" in wanted:
        try:
            from src.fetchers import google_trends
            n = db.upsert_series(con, google_trends.fetch(cfg))
            ok.append(f"trends ({n} points)")
        except Exception:
            failed.append("trends (pytrends is best-effort; Google throttles it)")
            traceback.print_exc()

    if "ga4" in wanted:
        try:
            from src.fetchers import ga4
            if ga4.available():
                n = db.upsert_series(con, ga4.fetch(cfg))
                ok.append(f"ga4 ({n} points)")
            else:
                print("[ga4] skipped — no credentials configured (see src/fetchers/ga4.py)")
        except Exception:
            failed.append("ga4")
            traceback.print_exc()

    if ok:
        db.touch_last_updated(con)
    print(f"\nfetched : {', '.join(ok) or 'nothing'}")
    print(f"failed  : {', '.join(failed) or 'nothing'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
