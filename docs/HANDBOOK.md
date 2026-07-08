# The PULSE-360 Handbook

*How the system works, end to end — every flow, every formula, every design
decision.*

---

## Table of contents

1. [What PULSE-360 is](#1-what-pulse-360-is)
2. [The big picture — one daily cycle](#2-the-big-picture--one-daily-cycle)
3. [The data model](#3-the-data-model)
4. [The fetch layer](#4-the-fetch-layer)
5. [Anchor normalization, explained](#5-anchor-normalization-explained)
6. [The transform layer — every metric](#6-the-transform-layer--every-metric)
7. [The Channel Health Score](#7-the-channel-health-score)
8. [The dashboard, component by component](#8-the-dashboard-component-by-component)
9. [Automation — the self-refreshing loop](#9-automation--the-self-refreshing-loop)
10. [Running & configuring it](#10-running--configuring-it)
11. [Design decisions & trade-offs](#11-design-decisions--trade-offs)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What PULSE-360 is

PULSE-360 answers one question continuously: **"How healthy is my brand's
marketing presence this week, compared to its competitors?"**

It does this by:

1. **Fetching** live signals from independent public sources (Google Trends,
   Wikipedia pageviews, GDELT world news) for a brand and its competitors.
2. **Normalizing** everything into one tiny schema, regardless of source.
3. **Transforming** raw daily values into four business signals: momentum,
   stability, sentiment, and share of voice.
4. **Scoring** each channel 0–100 with explicit weights — the **Channel
   Health Score** — and grading it A/B/C/D.
5. **Displaying** it all on an interactive dashboard.
6. **Repeating** automatically every day via GitHub Actions.

The methodology is not invented for this project. It is the weighted-signal
account-health model I built to run churn prevention across a ₹384 Mn
enterprise portfolio at Bharti Airtel, transplanted from *accounts* to
*marketing channels*. Same idea: many noisy signals → explicit weights → one
number an executive can act on.

---

## 2. The big picture — one daily cycle

```
 03:30 UTC — GitHub Actions wakes up
      │
      ▼
 run_fetch.py --sources wikipedia,news,trends
      │
      ├─► wikipedia_views.py ──► Wikimedia REST API      (per-article daily views)
      ├─► gdelt_news.py     ──► GDELT DOC 2.0 API        (news articles + VADER sentiment)
      └─► google_trends.py  ──► Google Trends (pytrends) (search interest, anchor-batched)
      │
      ▼
 data/pulse.db  (SQLite — upserts, so re-running is always safe)
      │
      ▼
 git commit + push  ("data: daily refresh YYYY-MM-DD")
      │
      ▼
 Dashboard (app.py) — recomputes every metric from raw data on load
```

Key properties of this cycle:

- **Idempotent.** Every write is an upsert keyed on `(source, entity, date)`
  or a stable mention id. Running the fetch twice changes nothing.
- **Isolated.** Each source runs in its own try/except. Google throttling
  pytrends never blocks the Wikipedia fetch.
- **Self-healing.** If a source fails today, tomorrow's run fills the gap —
  each fetch pulls a rolling 90-day window, not just "yesterday".
- **Nothing precomputed.** The database stores only raw observations. Every
  derived number on the dashboard is recomputed from raw data at load time,
  so a metric-definition change never requires a data migration.

---

## 3. The data model

Three tables. That's the whole schema (`src/db.py`):

```sql
series (source, entity, date, value)            -- PK: source+entity+date
mentions (id, source, entity, created,
          title, url, score, sentiment)         -- PK: id
meta (key, value)                               -- e.g. last_updated
```

**Why so small?** Because *every* source — search interest, pageviews,
article counts, even GA4 sessions if you plug it in — reduces to the same
observation: *"entity X had value V on date D according to source S."*
Fragmented sources become comparable the moment they share this shape. This
is the single most important design idea in the project: **the schema is the
unification layer.**

`mentions` exists because news/community items carry text worth keeping
(title, URL, sentiment) — but their *daily counts* are also folded into
`series` so the mentions channel participates in time-series metrics like
momentum.

---

## 4. The fetch layer

Each fetcher lives in `src/fetchers/` and follows one contract: *take the
config, return rows in the shared schema shape.* The runner
(`run_fetch.py`) wires them together and enforces isolation.

### 4.1 Wikipedia pageviews (`wikipedia_views.py`)

- **API:** Wikimedia REST `pageviews/per-article/.../daily/{start}/{end}` —
  official, free, keyless.
- **What it measures:** how many people opened the brand's Wikipedia article
  each day — a proxy for *research interest* (people actively looking the
  company up).
- **Quirks handled:**
  - Article titles must be **percent-encoded** (`AT&T` → `AT%26T`,
    parentheses too) — discovered the hard way with `Vi_(telecom)` 404s.
  - Each article is fetched **independently**: one missing article logs a
    warning and the others still load.
  - Data lags ~1 day, so the window ends at *yesterday*.

### 4.2 GDELT world news (`gdelt_news.py`)

- **API:** GDELT DOC 2.0 (`api.gdeltproject.org/api/v2/doc/doc`) — indexes
  global online news in near-realtime, free, keyless.
- **What it measures:** press coverage volume (daily article counts →
  `series`) and tone (each headline scored with VADER → `mentions`).
- **Quirks handled:**
  - GDELT enforces ~1 request per 5 seconds and blocks bursts hard, so the
    fetcher spaces entities **10s apart** and retries 429s with escalating
    backoff (12s/24s/36s), giving up per-entity, never per-run.
  - Mention ids are **`md5(url)`**, not Python's `hash()` — `hash()` is
    salted per process, which would re-insert every article as a duplicate
    on every run. (A real bug, caught in testing.)
  - `news_query` in the config exists because short brand names are
    ambiguous in news search ("Vi" would match everything). Each entity
    searches a distinctive phrase.

### 4.3 Google Trends (`google_trends.py`)

- **Client:** `pytrends` — unofficial, best-effort, aggressively throttled
  by Google. Treated as a bonus source: its failure never fails the run.
- **What it measures:** relative search interest (0–100 scale) — the
  broadest awareness signal of the three.
- **Quirks handled:** the 5-term comparison limit via **anchor
  normalization** (next section), per-batch retry with backoff, and partial
  tolerance — if batch 2 dies, batch 1's data is still kept.

### 4.4 Optional connectors

- **GA4** (`ga4.py`): activates only when `GA4_PROPERTY_ID` +
  `GOOGLE_APPLICATION_CREDENTIALS` are set. Pulls daily sessions →
  `("ga4", brand, date, sessions)`. The website becomes just another
  channel in the same scoring pipeline.
- **Reddit** (`reddit_mentions.py`): kept as a reference implementation, but
  Reddit returns 403 for most keyless scripted access since its 2023 API
  changes. Wire in OAuth via `praw` if you want community buzz.

**Extending:** any new source is ~40 lines: authenticate, fetch, emit
`(source, entity, date, value)` rows. Search Console, YouTube, Instagram —
same pattern.

---

## 5. Anchor normalization, explained

Google Trends compares **at most 5 terms per request**, and every request is
scaled to its own 0–100 where 100 = that request's peak. Two separate
requests are therefore **not comparable** — a "50" in one may be a "5" in
the other.

PULSE-360 tracks 7 brands, so it needs two requests on **one common scale**.
The trick: put the same brand (Airtel, the anchor) in *every* batch.

```
Batch 1:  [Airtel, Verizon, AT&T, T-Mobile, Vodafone]      → Airtel mean = 40
Batch 2:  [Airtel, Deutsche Telekom, China Mobile]         → Airtel mean = 80
```

Airtel's *real* interest didn't change between the two requests — only the
scale did. So batch 2 is running on a scale where numbers come out 2× larger
(80/40). To place batch 2 onto batch 1's scale, multiply all its values by:

```
scale = anchor_mean_batch1 / anchor_mean_batch2  =  40 / 80  =  0.5
```

Now Deutsche Telekom's "60" in batch 2 becomes a comparable "30" on the
common scale. This is a standard technique in search-interest analytics and
lives in `google_trends.py::fetch()`.

---

## 6. The transform layer — every metric

All in `src/transform.py`. Each metric is computed per `(source, entity)`
pair from raw series, fresh on every dashboard load.

### Momentum — *"is attention growing?"*

```
wow_change = (sum of last 7 days − sum of prior 7 days) / sum of prior 7 days
```

A ratio like +0.40 (+40%) or −0.15 (−15%). For scoring it is squashed to
0..1 with `0.5 * (1 + tanh(x / 0.25))`:

- Why tanh? Raw ratios are unbounded (a viral spike could be +900%) and
  would drown every other signal. Tanh maps any ratio smoothly into 0..1
  with 0% change = 0.5, ±25% ≈ 0.19/0.81, and extremes saturating near 0/1.

### Stability — *"is attention dependable or spiky?"*

```
stability = 1 − (std / mean) of the last 28 daily values     (clamped to 0..1)
```

`std/mean` is the coefficient of variation. A brand whose attention swings
wildly day-to-day scores low even if its average is high — the marketing
analogue of a volatile account in churn scoring: erratic engagement is risk.

### Sentiment — *"what's the tone of coverage?"*

Each news headline is scored with **VADER** (a lexicon-based sentiment model
tuned for short informal text) giving a compound score in −1..+1. The metric
is the mean compound of the last 30 days of mentions, mapped to 0..1 via
`(s + 1) / 2`. No mentions → neutral 0.5, so a silent channel is neither
rewarded nor punished.

### Share of voice — *"how much of the conversation is ours?"*

```
sov = entity's last-7-day volume / all entities' last-7-day volume   (per source)
```

The direct competitive-benchmark signal: shares across brands in one channel
sum to 1.

---

## 7. The Channel Health Score

```
score = 100 × ( 0.35 × momentum_n
              + 0.20 × stability
              + 0.25 × sentiment_n
              + 0.20 × sov )
```

| Grade | Range | Reading |
|---|---|---|
| **A** | ≥ 75 | Healthy — protect and amplify |
| **B** | 60–74 | Watch — trending well but not dominant |
| **C** | 45–59 | At risk — needs a plan |
| **D** | < 45 | Critical — losing the channel |

Design principles (identical to the account-health model this comes from):

- **Weights are explicit and live in config** (`config.yaml`,
  validated to sum to 1.0). Anyone can challenge or re-tune them — that's a
  feature. A score nobody can interrogate is a vibe, not a metric.
- **Momentum weighs heaviest (35%)** because direction beats position:
  a small brand growing 20% weekly is a better story than a giant slowly
  bleeding.
- **Every input is 0..1 before weighting** so no signal dominates by unit
  accident.
- The **brand's overall score** is the mean of its per-channel scores.

### Worked example

Airtel / Search Interest, with `wow = +0.075`, `stability = 0.81`,
no mentions (`sentiment_n = 0.5`), `sov = 0.31`:

```
momentum_n  = 0.5 × (1 + tanh(0.075 / 0.25)) = 0.5 × (1 + 0.291) = 0.646
score = 100 × (0.35×0.646 + 0.20×0.81 + 0.25×0.5 + 0.20×0.31)
      = 100 × (0.226 + 0.162 + 0.125 + 0.062) = 57.5  → 58 · C
```

---

## 8. The dashboard, component by component

`app.py`, top to bottom:

| Component | What it shows | How to read it |
|---|---|---|
| **KPI row** | Brand's overall score + one tile per channel with grade and WoW delta | The 5-second executive summary. Green/red arrows = momentum direction |
| **Attention over time** | Daily series for all brands, one channel at a time (radio switch) | Brand vs competitors trajectory. Colors are fixed per brand and never change when filters change |
| **Share of voice** | Horizontal bars: each brand's slice of the channel's last-7-day volume | Who owns the conversation this week |
| **Momentum (WoW)** | Diverging bars around zero, blue = growing, red = shrinking | Who is gaining vs losing, regardless of size |
| **Health score table** | Score per brand per channel + overall, with progress bars | The full competitive benchmark in one view |
| **Mentions table** | Recent headlines with engagement and sentiment (▲/▼/· labels) | The qualitative "why" behind the sentiment number |
| **Sidebar** | Methodology, data freshness, links | The provenance story |

Charting rules baked in: one axis per chart (never dual-axis), fixed
colorblind-validated palette assigned by entity order, direct value labels
on bars, unified hover crosshair on time series, and sentiment always shown
as a symbol + number, never color alone.

---

## 9. Automation — the self-refreshing loop

`.github/workflows/refresh-data.yml`:

```
Schedule: daily 03:30 UTC (09:00 IST)  +  manual "Run workflow" button
   1. checkout repo
   2. install Python 3.12 + requirements (pip cache)
   3. python run_fetch.py --sources wikipedia,news,trends
   4. git commit data/pulse.db  (only if it changed)  + push
```

Why **commit the database back to the repo**?

- The repo *is* the deployment artifact — Streamlit Cloud redeploys on every
  push, so the dashboard updates itself with zero infrastructure.
- Reviewers cloning the repo get real, current data instantly.
- The commit history becomes a free audit log of every data refresh.

This pattern (repo-as-database) is deliberately chosen for a portfolio-scale
project — see trade-offs below for when it stops being appropriate.

---

## 10. Running & configuring it

```bash
# one-time
pip install -r requirements.txt

# instant demo (no network, deterministic)
python run_fetch.py --demo
streamlit run app.py

# live data
python run_fetch.py                          # all keyless sources
python run_fetch.py --sources wikipedia      # just one
```

**Change the brands** in `config.yaml` — name, `wikipedia_article` (exact
article title), `news_query` (distinctive phrase for news search) — and
re-run the fetch. Everything downstream is brand-agnostic. Entity order
matters: the first entity is the "brand" (anchor + focal color), the rest
are competitors.

**Re-tune the score** by editing `health_weights` (must sum to 1.0 — the
config loader enforces it).

---

## 11. Design decisions & trade-offs

| Decision | Why | When to change it |
|---|---|---|
| **SQLite, committed to the repo** | Zero infra, instant reviewability, audit-log commits | Move to Postgres when data outgrows ~100 MB or writers multiply |
| **Streamlit over Dash/React** | Free hosted deploys, fastest iteration, Python-native | A React front end when the dashboard needs custom UX |
| **GitHub Actions over Airflow** | A daily fetch doesn't need a DAG engine | Airflow/Prefect when there are dependencies between jobs, retries with state, or many pipelines |
| **GDELT over Reddit for mentions** | Keyless, global, reliable; Reddit now demands OAuth | Add Reddit via praw if community tone matters more than press tone |
| **Raw data stored, metrics computed at read time** | Metric definitions can evolve without migrations | Precompute if load-time latency ever matters |
| **tanh squash on momentum** | Caps viral outliers without a hard cliff | Swap for winsorized z-scores if you need statistical formality |
| **VADER for sentiment** | Instant, lexicon-based, zero API cost, fine for headlines | An LLM scorer for nuance (sarcasm, mixed tone) at ~1000× the cost |

Known limitations, stated honestly:

- **pytrends is unofficial** — Google throttles it; some days the Trends
  channel just won't refresh. The design treats it as a bonus signal.
- **News volume ≠ marketing performance** — coverage is a proxy for
  awareness, not conversions. GA4 is the connector that closes that gap.
- **VADER reads headlines literally** — "Airtel slashes prices" scores
  negative ("slashes"). Directionally useful at volume, not per-article.
- **Wikipedia views skew English** — China Mobile's en-wiki views understate
  its domestic presence.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `trends: 429 TooManyRequests` | Google throttling pytrends | Wait — backoff retries are built in; it usually works within a few runs |
| `news: throttled, retrying…` then few mentions | GDELT burst-blocking | Normal; daily runs accumulate mentions cumulatively |
| `wikipedia: 404` for one brand | Wrong article title | Use the exact title from the article's URL (underscores, no encoding) |
| `reddit: 403 Blocked` | Reddit requires OAuth now | Expected; source is optional |
| Dashboard: "No data yet" | Fresh clone, no db | `python run_fetch.py --demo` |
| A brand missing from charts | Not in `config.yaml` | The app filters to configured entities by design |
| Weights error on startup | `health_weights` ≠ 1.0 | Fix the yaml; the loader enforces the invariant |

---

*Handbook for [PULSE-360](https://github.com/sougatshekhar97-cpu/pulse-360)
by Sougat Shekhar Hota ·
[Portfolio](https://sougatshekhar97-cpu.github.io/portfolio/)*
