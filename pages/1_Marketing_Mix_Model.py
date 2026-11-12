"""PULSE-MMM — Meridian-style Bayesian marketing mix model (Streamlit page).

Reads docs/mmm.json produced by `python run_mmm.py`; never runs the sampler
on page load, so it stays fast on Streamlit Community Cloud.
"""
import json
import math
import subprocess
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.config import ROOT

st.set_page_config(page_title="PULSE-360 · Marketing Mix Model", page_icon="📈", layout="wide")

SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9"}
PATH = ROOT / "docs" / "mmm.json"


def layout(fig, h=320, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK["secondary"]), **kw)
    fig.update_xaxes(gridcolor=INK["grid"]); fig.update_yaxes(gridcolor=INK["grid"])
    return fig


def rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


st.title("📈 Marketing Mix Model")

if not PATH.exists():
    st.warning("No model results yet. Fit the model once (about 2 minutes):")
    st.code("python run_mmm.py --demo")
    if st.button("Fit a quick demo model now (~30 s)"):
        with st.spinner("Sampling the posterior…"):
            subprocess.run([sys.executable, "run_mmm.py", "--demo", "--quick"], cwd=ROOT, check=True)
        st.rerun()
    st.stop()

M = json.loads(PATH.read_text(encoding="utf-8"))
CH = M["channels"]
NAMES = [c["name"] for c in CH]
COLOR = {n: SERIES[i] for i, n in enumerate(NAMES)}
CUR = M["currency"]
cr = lambda lakh: f"₹{lakh / 100:,.0f} Cr"   # 100 lakh = 1 crore

st.caption(
    f"Bayesian MMM in the style of Google Meridian · {len(M['geos'])} geos × {len(M['weeks'])} weeks · "
    f"adstock + Hill saturation + geo partial pooling · ROI as a counterfactual · fitted {M['generated']}"
)
if M["synthetic"]:
    st.info(
        "**Synthetic data with a known answer.** The input is a simulated Indian telecom brand across "
        f"{len(M['geos'])} circles with planted ROIs and planted confounders (Diwali, search intent, "
        "competitor promos). The ◆ markers show the true values, so you can check the model's work. "
        "Swap `data/mmm_input.csv` for real data and re-run `python run_mmm.py`.",
        icon="🧪",
    )

# ---------------------------------------------------------------- KPI row
cols = st.columns(len(CH))
for col, c in zip(cols, CH):
    r = c["roi"]
    with col:
        st.metric(c["name"], f"ROI {r['median']:.2f}",
                  f"90%: {r['lo']:.2f}–{r['hi']:.2f}", delta_color="off")
        st.caption(f"{c['saturation']} · next ₹1 returns ₹{c['mroi']['median']:.2f}"
                   + (f" · truth {c['truth']}" if "truth" in c else ""))

st.divider()

# ---------------------------------------------------------------- ROI intervals + optimizer
left, right = st.columns([1.1, 1])
with left:
    st.subheader("ROI with uncertainty")
    st.caption("Dot = posterior median, bar = 90% credible interval, ◆ = true value. "
               "A wide bar is the model saying *we don't know*.")
    fig = go.Figure()
    for c in reversed(CH):
        r = c["roi"]
        fig.add_trace(go.Scatter(x=[r["lo"], r["hi"]], y=[c["name"]] * 2, mode="lines",
                                 line=dict(color=COLOR[c["name"]], width=8), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r["median"]], y=[c["name"]], mode="markers",
                                 marker=dict(color="white", size=11, line=dict(color=COLOR[c["name"]], width=3)),
                                 showlegend=False,
                                 hovertemplate=f"{c['name']}: {r['median']:.2f} [{r['lo']:.2f}–{r['hi']:.2f}]<extra></extra>"))
        if "truth" in c:
            fig.add_trace(go.Scatter(x=[c["truth"]], y=[c["name"]], mode="markers",
                                     marker=dict(symbol="diamond", color=INK["primary"], size=10),
                                     showlegend=False, hovertemplate=f"truth {c['truth']}<extra></extra>"))
    fig.add_vline(x=1, line=dict(color=INK["muted"], dash="dot"))
    layout(fig, 300, xaxis=dict(title="₹ revenue per ₹1 spent", rangemode="tozero"))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Budget reallocation")
    o = M["optimizer"]
    st.caption(f"Same total budget ({cr(o['total'])} over the period); each channel may move "
               f"±{o['bounds']:.0%}. Channels the model is unsure about are held.")
    rows = [{"Channel": c["name"], "Current": cr(c["spend"]), "Recommended": cr(c["optimal_spend"]),
             "Change": "held · too uncertain" if c["held"] else f"{c['change_pct']:+.0%}",
             "Marginal ROI": f"{c['mroi']['median']:.2f}"} for c in CH]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    lift = o["lift"]
    st.metric("Expected extra revenue from reallocating",
              cr(lift["median"]), f"90%: {cr(lift['lo'])} – {cr(lift['hi'])}", delta_color="off")

st.divider()

# ---------------------------------------------------------------- response curves
st.subheader("Response curves — diminishing returns")
st.caption("Incremental revenue if each channel's whole plan were scaled from 0× to 2×. "
           "Band = 90% interval. ● current spend, ○ recommended.")
mults = M["curves"]["mults"]
cols = st.columns(len(CH))
for col, c in zip(cols, CH):
    cv = M["curves"][c["name"]]
    xs = [m * c["spend"] / 100 for m in mults]
    col_hex = COLOR[c["name"]]
    fig = go.Figure([
        go.Scatter(x=xs + xs[::-1], y=[v / 100 for v in cv["hi"]] + [v / 100 for v in cv["lo"][::-1]],
                   fill="toself", fillcolor=rgba(col_hex, 0.18), line=dict(width=0), hoverinfo="skip"),
        go.Scatter(x=xs, y=[v / 100 for v in cv["median"]], line=dict(color=col_hex, width=2.5),
                   hovertemplate="spend ₹%{x:,.0f} Cr → +₹%{y:,.0f} Cr<extra></extra>"),
    ])
    i1 = mults.index(1.0)
    fig.add_trace(go.Scatter(x=[c["spend"] / 100], y=[cv["median"][i1] / 100], mode="markers",
                             marker=dict(color=col_hex, size=10), hoverinfo="skip"))
    om = 1 + c["change_pct"]
    if abs(c["change_pct"]) > 0.005:
        y_opt = float(np.interp(om, mults, cv["median"])) / 100
        fig.add_trace(go.Scatter(x=[c["optimal_spend"] / 100], y=[y_opt], mode="markers",
                                 marker=dict(color="white", size=10, line=dict(color=col_hex, width=2.5)),
                                 hoverinfo="skip"))
    layout(fig, 240, showlegend=False, title=dict(text=c["name"], font=dict(size=14)),
           xaxis=dict(title="spend, ₹ Cr"), yaxis=dict(title="incremental, ₹ Cr" if c is CH[0] else None))
    with col:
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------- causal: naive vs controlled
if any("roi_without" in c for c in CH):
    st.subheader("Why the causal controls matter")
    dropped = ", ".join(M["compare_without"])
    st.caption(f"Same model re-fitted **without `{dropped}`** (Google query volume, a measure of "
               "existing purchase intent). Search ads show up when people already want the product, "
               "so without it the model credits search with demand that existed anyway.")
    fig = go.Figure()
    fig.add_trace(go.Bar(name=f"Without {dropped}", x=NAMES, y=[c["roi_without"]["median"] for c in CH],
                         marker_color=INK["muted"],
                         text=[f"{c['roi_without']['median']:.1f}" for c in CH], textposition="outside"))
    fig.add_trace(go.Bar(name="With causal controls", x=NAMES, y=[c["roi"]["median"] for c in CH],
                         marker_color=SERIES[0],
                         text=[f"{c['roi']['median']:.1f}" for c in CH], textposition="outside"))
    if all("truth" in c for c in CH):
        fig.add_trace(go.Scatter(name="Truth", x=NAMES, y=[c["truth"] for c in CH], mode="markers",
                                 marker=dict(symbol="diamond", size=11, color=INK["primary"])))
    layout(fig, 320, barmode="group", yaxis=dict(type="log", title="ROI (log scale)"),
           legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

# ---------------------------------------------------------------- prior vs posterior
st.subheader("Prior vs posterior — did the data actually speak?")
st.caption("Grey curve = prior (belief before this data). Bars = posterior. If the two look alike, "
           "the answer came from the prior, not the data.")
cols = st.columns(len(CH))
for col, c in zip(cols, CH):
    h = c["posterior_hist"]; e = h["edges"]
    mids = [(a + b) / 2 for a, b in zip(e, e[1:])]
    width = e[1] - e[0]
    total = sum(h["counts"]) * width
    mu, sd = c["prior"]["mean"], c["prior"]["sd"]
    lo, hi = min(e[0], mu - 2.5 * sd, 0), max(e[-1], mu + 2.5 * sd)
    xs = [lo + (hi - lo) * i / 120 for i in range(121)]
    pdf = [math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)) for x in xs]
    fig = go.Figure([
        go.Bar(x=mids, y=[n / total for n in h["counts"]], width=width,
               marker_color=rgba(COLOR[c["name"]], 0.75), hoverinfo="skip"),
        go.Scatter(x=xs, y=pdf, line=dict(color=INK["muted"], width=2, dash="dot"), hoverinfo="skip"),
    ])
    layout(fig, 200, showlegend=False, title=dict(text=c["name"], font=dict(size=14)),
           xaxis=dict(title="ROI"), yaxis=dict(showticklabels=False))
    with col:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Prior {mu} ± {sd} ({c['prior']['source']}). "
                   f"Posterior is {c['prior_dominance']:.0%} as wide.")

st.divider()

# ---------------------------------------------------------------- decomposition
st.subheader("What drove revenue, week by week")
st.caption("National revenue split into baseline (intercepts, trend, season, controls) and each channel's "
           "incremental contribution. Black line = actual.")
D = M["decomposition"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=M["weeks"], y=[v / 100 for v in D["baseline"]], name="Baseline",
                         stackgroup="one", line=dict(width=0), fillcolor=rgba(INK["muted"], 0.25)))
for n in NAMES:
    fig.add_trace(go.Scatter(x=M["weeks"], y=[v / 100 for v in D[n]], name=n, stackgroup="one",
                             line=dict(width=0), fillcolor=rgba(COLOR[n], 0.8)))
fig.add_trace(go.Scatter(x=M["weeks"], y=[v / 100 for v in D["actual"]], name="Actual",
                         line=dict(color=INK["primary"], width=1.5)))
base_min = min(D["baseline"]) / 100
layout(fig, 360, hovermode="x unified", legend=dict(orientation="h", y=1.1),
       yaxis=dict(title="₹ Cr / week", range=[base_min * 0.9, max(D["actual"]) / 100 * 1.03]))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- caveats + diagnostics
st.subheader("Read before acting")
for w in M["warnings"]:
    st.warning(w)
st.markdown(
    "- **Still observational data.** The causal reading is only as good as the controls: miss a "
    "confounder and the intervals will be confidently wrong.\n"
    "- **Priors can dominate.** Check the prior-vs-posterior panel; a posterior that just copies its "
    "prior isn't evidence.\n"
    "- **Best practice:** feed periodic lift-test results back in as ROI priors (`config.yaml → mmm → "
    "roi_prior`). The model and the experiments keep each other honest."
)

with st.expander("Model diagnostics"):
    fit = M["fit"]
    a, b, c3 = st.columns(3)
    a.metric("R² (national weekly)", f"{fit['r2']:.3f}")
    b.metric("MAPE", f"{fit['mape']:.2%}")
    c3.metric("Max R-hat", f"{fit['max_rhat']:.3f}", "< 1.1 = chains agree", delta_color="off")
    st.dataframe([{"Channel": c["name"], "Adstock decay": c["alpha"], "Hill half-saturation (× median spend)": c["K"],
                   "R-hat": c["rhat"], "MH acceptance": f"{M['sampler']['acceptance'][c['name']]:.0%}"}
                  for c in CH], hide_index=True, use_container_width=True)
    st.write(f"Controls (confounders): `{', '.join(M['controls'])}` + trend + 2 annual Fourier pairs + "
             f"one intercept per geo. Sampler: {M['sampler'].get('chains')} chains × "
             f"{M['sampler'].get('draws')} draws after {M['sampler'].get('warmup')} warm-up.")
