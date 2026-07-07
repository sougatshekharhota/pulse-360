"""Google Analytics 4 connector (optional — needs credentials).

Activates only when both are present:
  - env var GA4_PROPERTY_ID
  - env var GOOGLE_APPLICATION_CREDENTIALS pointing to a service-account JSON
    that has Viewer access on the GA4 property

Install the extra dependency first:  pip install google-analytics-data

Pulls daily sessions for the property and stores them as the "ga4" source
under the brand entity, so the website channel joins the same health-score
pipeline as the public sources.
"""
import os


def available() -> bool:
    return bool(os.environ.get("GA4_PROPERTY_ID")) and bool(
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )


def fetch(cfg: dict) -> list[tuple]:
    if not available():
        return []

    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest

    prop = os.environ["GA4_PROPERTY_ID"]
    lookback = int(cfg.get("lookback_days", 90))
    client = BetaAnalyticsDataClient()
    request = RunReportRequest(
        property=f"properties/{prop}",
        dimensions=[Dimension(name="date")],
        metrics=[Metric(name="sessions")],
        date_ranges=[DateRange(start_date=f"{lookback}daysAgo", end_date="yesterday")],
    )
    response = client.run_report(request)

    brand = cfg["brand"]["name"]
    rows: list[tuple] = []
    for r in response.rows:
        ymd = r.dimension_values[0].value  # YYYYMMDD
        iso = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
        rows.append(("ga4", brand, iso, float(r.metric_values[0].value)))
    return rows
