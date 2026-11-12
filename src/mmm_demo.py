"""Synthetic geo-level MMM dataset with a KNOWN ground truth.

12 Indian telecom circles x 156 weeks. Built to contain every trap the MMM
has to survive, so you can check the engine recovers the planted ROIs:

  * Diwali lifts demand AND every channel's spend      -> confounder (festive)
  * latent purchase intent drives sales AND search spend -> confounder (gqv)
  * competitor promos cut sales AND trigger defensive TV -> confounder
  * July-2024 tariff hike moves revenue                  -> control (price_index)
  * TV and YouTube share a national flighting calendar   -> collinearity
  * small circles are noisier than big ones              -> geo pooling helps

Ground truth uses the same adstock/Hill family the model assumes, so this is
a recovery test of the inference — not proof the functional form is right
for real data.
"""
import json

import numpy as np
import pandas as pd

from .mmm import adstock, hill

CIRCLES = [  # (circle, population in millions — rounded, illustrative)
    ("Mumbai", 21), ("Delhi", 32), ("Karnataka", 68), ("Tamil Nadu", 77),
    ("Maharashtra & Goa", 95), ("Gujarat", 70), ("UP East", 130), ("West Bengal", 100),
    ("Andhra Pradesh", 90), ("Kerala", 35), ("Rajasthan", 80), ("Punjab", 30),
]
CHANNELS = {"TV": "spend_tv", "YouTube": "spend_youtube",
            "Meta": "spend_meta", "Paid Search": "spend_search"}
CONTROLS = ["festive", "gqv", "price_index", "promo", "competitor_promo"]

# planted truth: ROI, adstock decay, Hill half-saturation (in median-spend units)
TRUTH = {
    "TV":          {"roi": 1.4, "alpha": 0.60, "K": 0.45},   # near saturated
    "YouTube":     {"roi": 2.2, "alpha": 0.40, "K": 2.50},   # room to grow
    "Meta":        {"roi": 1.7, "alpha": 0.20, "K": 1.20},
    "Paid Search": {"roi": 1.1, "alpha": 0.05, "K": 1.00},
}
DIWALI = ["2023-11-12", "2024-11-01", "2025-10-21"]


def _blocks(rng, T, p_on, length):
    """On/off flighting in blocks of `length` weeks."""
    on = rng.uniform(size=T // length + 1) < p_on
    return np.repeat(on, length)[:T].astype(float)


def generate(seed: int = 42, weeks: int = 156, start: str = "2023-01-02"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=weeks, freq="W-MON")
    G, T = len(CIRCLES), weeks
    pop = np.array([p for _, p in CIRCLES], float)
    t = np.arange(T)

    # ---- calendar + market drivers (national)
    days = np.array([(dates - pd.Timestamp(d)).days for d in DIWALI])
    festive = np.exp(-0.5 * (days / 12.0) ** 2).max(0)
    ipl = dates.month.isin([3, 4, 5]).astype(float)
    price = np.where(dates >= "2024-07-03", 1.12, 1.0) + rng.normal(0, 0.005, T)
    promo = (rng.uniform(size=T) < 0.12).astype(float)
    comp = _blocks(rng, T, 0.15, 5)

    # ---- latent intent per circle (AR(1)), observed only through query volume
    q = np.zeros((G, T))
    for i in range(1, T):
        q[:, i] = 0.8 * q[:, i - 1] + rng.normal(0, 0.025, G)
    gqv = 100 * (1 + q + 0.3 * festive) + rng.normal(0, 0.3, (G, T))

    # ---- spend per capita (₹ lakh per million people per week)
    gm = rng.lognormal(0, 0.25, (4, G))[:, :, None]
    flight = _blocks(rng, T, 0.6, 4)
    own = lambda p, l: np.stack([_blocks(rng, T, p, l) for _ in range(G)])
    tv = 0.7 * gm[0] * (0.15 + 0.6 * flight + 0.5 * own(0.5, 3)) \
        * (1 + 1.0 * festive + 0.4 * ipl + 0.3 * comp) * rng.lognormal(0, 0.2, (G, T))
    yt = 0.3 * gm[1] * (0.4 + 0.5 * flight + 0.6 * own(0.5, 3)) \
        * (1 + 0.8 * festive) * rng.lognormal(0, 0.25, (G, T))
    meta = 0.25 * gm[2] * (0.5 + 0.8 * own(0.5, 2)) \
        * (1 + 0.6 * festive + 0.4 * promo) * rng.lognormal(0, 0.3, (G, T))
    search = 0.18 * gm[3] * np.exp(4 * q) * (1 + 0.5 * festive) * rng.lognormal(0, 0.1, (G, T))
    spend_pc = np.stack([tv, yt, meta, search])                 # (C, G, T)
    spend = spend_pc * pop[None, :, None]

    # ---- baseline revenue per capita (₹ lakh per million per week)
    # ponytail: drivers are additive per capita (shared across circles), matching
    # the model's common control coefficients; Meridian also offers geo-varying
    # control effects — the upgrade path if real circles react differently.
    geo_level = 40 * rng.lognormal(0, 0.15, G)[:, None]
    base_pc = geo_level + 40 * (
        0.06 * t / 52 + 0.08 * festive
        + 0.03 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)
        + 0.5 * (price - 1) + 0.03 * promo - 0.04 * comp + q)

    # ---- planted media effects, calibrated so counterfactual ROI == TRUTH
    med = np.array([np.median(s) for s in spend_pc])
    media = np.zeros((4, G, T))
    for c, name in enumerate(CHANNELS):
        tr = TRUTH[name]
        H = hill(adstock(spend_pc[c] / med[c], tr["alpha"]), tr["K"])
        geo_b = rng.lognormal(0, 0.2, G)[:, None]
        unit = (pop[:, None] * geo_b * H).sum() / spend[c].sum()
        media[c] = pop[:, None] * geo_b * H * tr["roi"] / unit

    noise = base_pc * 0.01 * np.sqrt(pop.mean() / pop)[:, None] * rng.normal(size=(G, T))
    revenue = pop[:, None] * (base_pc + noise) + media.sum(0)

    # ---- long CSV
    rows = []
    for g, (circle, p) in enumerate(CIRCLES):
        for i in range(T):
            rows.append({
                "geo": circle, "week": dates[i].strftime("%Y-%m-%d"), "population": p,
                "revenue": round(revenue[g, i], 2),
                **{col: round(spend[c, g, i], 3) for c, col in enumerate(CHANNELS.values())},
                "festive": round(festive[i], 4), "gqv": round(gqv[g, i], 2),
                "price_index": round(price[i], 4), "promo": int(promo[i]),
                "competitor_promo": int(comp[i]),
            })
    truth = {name: {**TRUTH[name],
                    "incremental_revenue": round(float(media[c].sum()), 1),
                    "spend": round(float(spend[c].sum()), 1)}
             for c, name in enumerate(CHANNELS)}
    return pd.DataFrame(rows), truth


def write(csv_path, truth_path, seed: int = 42):
    df, truth = generate(seed)
    df.to_csv(csv_path, index=False)
    with open(truth_path, "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2)
    return df, truth
