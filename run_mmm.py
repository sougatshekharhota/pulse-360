"""Fit PULSE-MMM and export docs/mmm.json.

    python run_mmm.py --demo     # regenerate the synthetic dataset, then fit
    python run_mmm.py            # fit whatever CSV config.yaml points at
    python run_mmm.py --quick    # fewer draws (smoke test)

The JSON is read by both the Streamlit page and the static GitHub Pages
dashboard, so neither needs to run the sampler.
"""
import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

from src import mmm
from src.config import ROOT, load_config

OUT = ROOT / "docs" / "mmm.json"
CURVE_MULTS = np.round(np.arange(0, 2.0001, 0.05), 2)
OPT_MULTS = np.round(np.arange(0.6, 1.4001, 0.01), 2)
UNCERTAIN = 1.2  # (90% width / median) above this = "we don't know" -> don't move budget


def q(x, p=(5, 50, 95)):
    lo, mid, hi = np.percentile(x, p)
    return {"lo": round(float(lo), 3), "median": round(float(mid), 3), "hi": round(float(hi), 3)}


def thinned(f: mmm.Fit, per_chain: int):
    """Indices kept per chain — preserves chain structure for R-hat."""
    keep = np.linspace(0, f.theta.shape[1] - 1, per_chain).astype(int)
    return f.theta[:, keep], f.alpha[:, keep], f.K[:, keep]


def roi_draws(f: mmm.Fit, per_chain: int = 150) -> np.ndarray:
    th, a, k = thinned(f, per_chain)
    ch, n = th.shape[:2]
    return np.array([[mmm.counterfactual_roi(f, th[i, j], a[i, j], k[i, j]) for j in range(n)]
                     for i in range(ch)])                                 # (chains, n, C)


def analyse(f: mmm.Fit, cfg: dict, truth: dict | None, compare: dict | None) -> dict:
    d = f.data
    C = len(d.channels)
    spend = d.spend.sum(axis=(1, 2))
    R = roi_draws(f)
    Rf = R.reshape(-1, C)

    th, a, k = (x.reshape(-1, *x.shape[2:]) for x in thinned(f, 60))
    # response curves + marginal ROI, per draw
    curves = np.array([[mmm.response_curve(f, th[i], a[i], k[i], c, CURVE_MULTS)
                        for c in range(C)] for i in range(len(th))])       # (D, C, M)
    fine = np.array([[mmm.response_curve(f, th[i], a[i], k[i], c, OPT_MULTS)
                      for c in range(C)] for i in range(len(th))])
    i100, i101 = list(OPT_MULTS).index(1.0), list(OPT_MULTS).index(1.01)
    mroi = (fine[:, :, i101] - fine[:, :, i100]) / (0.01 * spend)

    # budget optimizer: total fixed, each channel within ±bounds — except
    # channels whose ROI interval is too wide to act on, which are held.
    bounds = float(cfg.get("optimizer_bounds", 0.3))
    lo, mid, hi = np.percentile(Rf, [5, 50, 95], axis=0)
    uncertain = (hi - lo) / np.maximum(np.abs(mid), 1e-9) > UNCERTAIN
    alloc = mmm.optimize_budget(fine.mean(0), OPT_MULTS, spend, np.where(uncertain, 0.0, bounds))
    opt_m = alloc / spend
    lift = np.array([sum(np.interp(opt_m[c], OPT_MULTS, fine[i, c]) - fine[i, c, i100]
                         for c in range(C)) for i in range(len(fine))])

    # decomposition: national weekly actual vs baseline + each channel
    media = np.mean([mmm.media_revenue(f, th[i], a[i], k[i]) for i in range(len(th))], 0)
    pred = np.mean([mmm.predict_revenue(f, th[i], a[i], k[i]) for i in range(len(th))], 0)
    nat_actual, nat_pred = d.revenue.sum(0), pred.sum(0)
    nat_media = media.sum(1)                                                # (C, T)
    r2 = 1 - ((nat_actual - nat_pred) ** 2).sum() / ((nat_actual - nat_actual.mean()) ** 2).sum()
    mape = float(np.mean(np.abs(nat_actual - nat_pred) / nat_actual))

    # diagnostics
    rh = {n: round(max(mmm.rhat(R[:, :, c]), mmm.rhat(f.alpha[:, :, c]), mmm.rhat(f.K[:, :, c])), 3)
          for c, n in enumerate(d.channels)}
    nat_spend = d.spend.sum(1)                                              # (C, T)
    corr = np.corrcoef(nat_spend)

    warnings = []
    pri = cfg["channels"]
    rows = []
    for c, name in enumerate(d.channels):
        roi = q(Rf[:, c]); mr = q(mroi[:, c])
        prior = pri[c]["roi_prior"]
        dominance = float(Rf[:, c].std() / prior["sd"])
        # label by the return on the NEXT rupee, not the average one
        sat = ("Near saturated" if mr["median"] < 0.7
               else "Diminishing" if mr["median"] < 1.2 else "Room to grow")
        hist, edges = np.histogram(Rf[:, c], bins=30)
        if rh[name] > 1.1:
            warnings.append(f"{name}: R-hat {rh[name]} > 1.1 — chains disagree, run longer before trusting it.")
        if dominance > 0.7:
            warnings.append(f"{name}: posterior is {dominance:.0%} as wide as the prior — the data barely "
                            f"moved it, so this ROI mostly reflects the prior ({prior['source']}).")
        if uncertain[c]:
            warnings.append(f"{name}: 90% interval {roi['lo']:.1f}–{roi['hi']:.1f} is too wide to move "
                            f"budget on, so the optimizer holds it. Vary its spend deliberately or run a lift test.")
        row = {
            "name": name, "spend": round(float(spend[c]), 1),
            "roi": roi, "mroi": mr, "saturation": sat,
            "contribution": round(float(Rf[:, c].mean() * spend[c]), 1),
            "alpha": round(float(np.median(f.alpha[:, :, c])), 3),
            "K": round(float(np.median(f.K[:, :, c])), 3),
            "rhat": rh[name], "prior": prior, "prior_dominance": round(dominance, 3),
            "posterior_hist": {"edges": [round(float(e), 3) for e in edges], "counts": hist.tolist()},
            "optimal_spend": round(float(alloc[c]), 1),
            "change_pct": round(float(opt_m[c] - 1), 4),
            "held": bool(uncertain[c]),
        }
        if truth and name in truth:
            row["truth"] = truth[name]["roi"]
        if compare:
            row["roi_without"] = compare[name]
        rows.append(row)

    for i in range(C):
        for j in range(i + 1, C):
            if abs(corr[i, j]) > 0.7:
                warnings.append(f"{d.channels[i]} and {d.channels[j]} spend move together "
                                f"(r = {corr[i, j]:.2f}); their split relies on geo-level differences.")

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "currency": cfg.get("currency", ""),
        "synthetic": bool(truth),
        "geos": d.geos, "weeks": [w.strftime("%Y-%m-%d") for w in d.weeks],
        "controls": d.control_names,
        "compare_without": cfg.get("compare_without", []) if compare else [],
        "sampler": {**cfg.get("sampler", {}), "acceptance": dict(zip(d.channels, np.round(f.accept, 3).tolist()))},
        "fit": {"r2": round(float(r2), 4), "mape": round(mape, 4), "max_rhat": max(rh.values())},
        "channels": rows,
        "curves": {"mults": CURVE_MULTS.tolist(),
                   **{name: {"median": np.median(curves[:, c], 0).round(1).tolist(),
                             "lo": np.percentile(curves[:, c], 5, 0).round(1).tolist(),
                             "hi": np.percentile(curves[:, c], 95, 0).round(1).tolist()}
                      for c, name in enumerate(d.channels)}},
        "optimizer": {"bounds": bounds, "total": round(float(spend.sum()), 1), "lift": q(lift)},
        "decomposition": {"actual": nat_actual.round(1).tolist(), "predicted": nat_pred.round(1).tolist(),
                          "baseline": (nat_pred - nat_media.sum(0)).round(1).tolist(),
                          **{name: nat_media[c].round(1).tolist() for c, name in enumerate(d.channels)}},
        "correlation": np.round(corr, 3).tolist(),
        "warnings": warnings,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="regenerate the synthetic dataset first")
    ap.add_argument("--quick", action="store_true", help="short chains (smoke test)")
    args = ap.parse_args()

    cfg = load_config()["mmm"]
    data_path, truth_path = ROOT / cfg["data"], ROOT / cfg.get("truth", "data/mmm_truth.json")
    if args.demo:
        from src.mmm_demo import write
        write(data_path, truth_path)
        print(f"wrote synthetic data -> {data_path.relative_to(ROOT)}")
    truth = json.loads(truth_path.read_text(encoding="utf-8")) if truth_path.exists() else None

    chans = {c["name"]: c["column"] for c in cfg["channels"]}
    prior = [[c["roi_prior"]["mean"], c["roi_prior"]["sd"]] for c in cfg["channels"]]
    s = dict(cfg.get("sampler", {}))
    if args.quick:
        s.update(chains=2, warmup=200, draws=200)
    kw = dict(chains=s.get("chains", 4), warmup=s.get("warmup", 800),
              draws=s.get("draws", 800), seed=s.get("seed", 7))

    d = mmm.load_csv(data_path, chans, cfg["controls"])
    print(f"fitting {len(d.geos)} geos x {len(d.weeks)} weeks x {len(d.channels)} channels ...")
    t0 = time.time()
    f = mmm.fit(d, prior, **kw)
    print(f"  sampled in {time.time() - t0:.0f}s")

    compare = None
    drop = [c for c in cfg.get("compare_without", []) if c in cfg["controls"]]
    if drop:
        print(f"re-fitting without {drop} for the naive comparison ...")
        d2 = mmm.load_csv(data_path, chans, [c for c in cfg["controls"] if c not in drop])
        f2 = mmm.fit(d2, prior, **{**kw, "chains": 2}, verbose=False)
        R2 = roi_draws(f2, 100).reshape(-1, len(chans))
        compare = {n: q(R2[:, c]) for c, n in enumerate(chans)}

    out = analyse(f, cfg, truth, compare)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size // 1024} KB)")
    for r in out["channels"]:
        t = f"  truth {r['truth']}" if "truth" in r else ""
        print(f"  {r['name']:<12} ROI {r['roi']['median']:.2f} "
              f"[{r['roi']['lo']:.2f}–{r['roi']['hi']:.2f}]  {r['saturation']:<14} "
              f"opt {r['change_pct']:+.0%}{t}")
    print(f"  optimizer lift: {out['optimizer']['lift']}")
    for w in out["warnings"]:
        print("  ! " + w)


if __name__ == "__main__":
    main()
