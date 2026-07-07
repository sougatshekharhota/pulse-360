"""Derived metrics + the Channel Health Score.

The score is a direct transplant of the weighted-signal account health model
I built for enterprise account management: several noisy signals, each
normalized to 0..1, combined with explicit weights into one 0-100 number
that an executive can act on.

Components per (source, entity):
  momentum        last 7-day volume vs the prior 7 days, squashed to 0..1
  stability       1 - coefficient of variation of the last 28 daily values
  sentiment       mean VADER compound of the last 30 days of mentions, mapped to 0..1
  share_of_voice  entity's share of the source's last-7-day total volume
"""
import math

import pandas as pd

SOURCE_LABELS = {
    "trends": "Search Interest",
    "wikipedia": "Knowledge Interest",
    "news": "News Coverage",
    "reddit": "Community Buzz",
    "ga4": "Website Traffic",
}


def _squash(x: float, scale: float = 0.25) -> float:
    """Map an unbounded ratio-change (e.g. +0.4 = +40%) to 0..1 via tanh."""
    return 0.5 * (1 + math.tanh(x / scale))


def weekly_momentum(series: pd.DataFrame) -> pd.DataFrame:
    """WoW % change of 7-day volume per (source, entity)."""
    out = []
    for (source, entity), g in series.groupby(["source", "entity"]):
        g = g.sort_values("date")
        last7 = g.tail(7)["value"].sum()
        prev7 = g.iloc[-14:-7]["value"].sum() if len(g) >= 14 else float("nan")
        change = (last7 - prev7) / prev7 if prev7 and prev7 > 0 else 0.0
        out.append({"source": source, "entity": entity,
                    "last7": last7, "prev7": prev7, "wow_change": change})
    return pd.DataFrame(out)


def stability(series: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (source, entity), g in series.groupby(["source", "entity"]):
        vals = g.sort_values("date").tail(28)["value"]
        mean = vals.mean()
        cv = (vals.std() / mean) if mean else 1.0
        out.append({"source": source, "entity": entity,
                    "stability": max(0.0, min(1.0, 1.0 - cv))})
    return pd.DataFrame(out)


def share_of_voice(series: pd.DataFrame) -> pd.DataFrame:
    mom = weekly_momentum(series)
    mom["sov"] = mom.groupby("source")["last7"].transform(
        lambda s: s / s.sum() if s.sum() else 0.0
    )
    return mom[["source", "entity", "sov"]]


def sentiment_by_entity(mentions: pd.DataFrame) -> pd.DataFrame:
    if mentions.empty:
        return pd.DataFrame(columns=["entity", "sentiment"])
    cutoff = mentions["created"].max() - pd.Timedelta(days=30)
    recent = mentions[mentions["created"] >= cutoff]
    return recent.groupby("entity")["sentiment"].mean().reset_index()


def health_scores(series: pd.DataFrame, mentions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per (source, entity) score 0-100 + grade, plus the weighted inputs."""
    if series.empty:
        return pd.DataFrame()
    w = cfg["health_weights"]

    df = weekly_momentum(series).merge(stability(series), on=["source", "entity"])
    df = df.merge(share_of_voice(series), on=["source", "entity"])
    sent = sentiment_by_entity(mentions).rename(columns={"sentiment": "sent_raw"})
    df = df.merge(sent, on="entity", how="left")
    df["sent_raw"] = df["sent_raw"].fillna(0.0)

    df["momentum_n"] = df["wow_change"].apply(_squash)
    df["sentiment_n"] = (df["sent_raw"] + 1) / 2

    df["score"] = 100 * (
        w["momentum"] * df["momentum_n"]
        + w["stability"] * df["stability"]
        + w["sentiment"] * df["sentiment_n"]
        + w["share_of_voice"] * df["sov"]
    )
    df["score"] = df["score"].round(0).astype(int)
    df["grade"] = df["score"].apply(grade)
    df["channel"] = df["source"].map(SOURCE_LABELS).fillna(df["source"])
    return df


def grade(score: float) -> str:
    if score >= 75:
        return "A"
    if score >= 60:
        return "B"
    if score >= 45:
        return "C"
    return "D"


GRADE_STATUS = {  # reserved status roles — shipped with the letter, never color alone
    "A": ("good", "#0ca30c"),
    "B": ("warning", "#fab219"),
    "C": ("serious", "#ec835a"),
    "D": ("critical", "#d03b3b"),
}


def brand_overall(scores: pd.DataFrame, brand: str) -> tuple[int, str]:
    mine = scores[scores["entity"] == brand]
    if mine.empty:
        return 0, "D"
    avg = int(round(mine["score"].mean()))
    return avg, grade(avg)
