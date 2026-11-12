"""PULSE-MMM — a Meridian-style Bayesian marketing mix model in pure NumPy.

Inspired by Google Meridian (no code taken from it). The same ideas, small
enough to read in one sitting:

  * Adstock (carryover)      geometric decay, normalized weights, max lag L
  * Hill saturation          h = x / (x + K)   (diminishing returns)
  * Geo hierarchy            beta[g,c] = beta_bar[c] + delta[g,c],
                             delta ~ N(0, (rho_c * prior_beta_c)^2)  -> partial pooling
  * ROI priors               prior is placed on ROI (what experiments measure),
                             mapped to beta through the current adstock/Hill shape
  * Causal controls          confounders (festive season, query volume, price...)
                             enter as controls; mediators are deliberately left out
  * Counterfactual ROI       revenue(actual spend) - revenue(channel zeroed)

Inference is a partially collapsed Gibbs sampler:
  1. (alpha_c, K_c) per channel via Metropolis-Hastings, with every linear
     coefficient integrated out analytically (Gaussian marginal likelihood)
  2. all linear coefficients jointly from their exact Gaussian conditional
  3. noise variance and pooling scales from their inverse-gamma conditionals
No TensorFlow, no compiler: numpy only, a full fit takes well under a minute.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

L_MAX = 8  # adstock window in weeks


# ---------------------------------------------------------------- transforms
def adstock(x: np.ndarray, alpha: float, L: int = L_MAX) -> np.ndarray:
    """Geometric carryover along the last axis; weights sum to 1, so total
    effect is preserved and only its timing spreads out."""
    w = alpha ** np.arange(L)
    w /= w.sum()
    out = np.zeros_like(x, dtype=float)
    T = x.shape[-1]
    for lag in range(min(L, T)):
        out[..., lag:] += w[lag] * x[..., : T - lag]
    return out


def hill(x: np.ndarray, K: float) -> np.ndarray:
    """Hill saturation with slope 1: half the maximum effect at x = K.
    ponytail: slope fixed at 1 for identifiability with ~2k rows; free slope
    needs either more data or a tight prior (Meridian samples it)."""
    return x / (x + K)


# ---------------------------------------------------------------- data
@dataclass
class MMMData:
    geos: list[str]
    weeks: pd.DatetimeIndex
    pop: np.ndarray            # (G,)
    revenue: np.ndarray        # (G, T)
    spend: np.ndarray          # (C, G, T)
    channels: list[str]
    controls: np.ndarray       # (P, G, T)
    control_names: list[str]


def load_csv(path, channel_cols: dict[str, str], control_cols: list[str]) -> MMMData:
    """Long CSV: one row per (geo, week) with population, revenue, spend
    columns and control columns. A single geo (national data) also works."""
    df = pd.read_csv(path, parse_dates=["week"]).sort_values(["geo", "week"])
    geos = list(dict.fromkeys(df["geo"]))
    weeks = pd.DatetimeIndex(sorted(df["week"].unique()))
    G, T = len(geos), len(weeks)
    if len(df) != G * T:
        raise ValueError(f"expected a full geo x week panel ({G}x{T}), got {len(df)} rows")
    grid = lambda col: df[col].to_numpy(float).reshape(G, T)
    return MMMData(
        geos=geos, weeks=weeks,
        pop=df.groupby("geo", sort=False)["population"].first().reindex(geos).to_numpy(float),
        revenue=grid("revenue"),
        spend=np.stack([grid(c) for c in channel_cols.values()]),
        channels=list(channel_cols),
        controls=np.stack([grid(c) for c in control_cols]) if control_cols else np.zeros((0, G, T)),
        control_names=list(control_cols),
    )


# ---------------------------------------------------------------- model
@dataclass
class Fit:
    data: MMMData
    roi_prior: np.ndarray            # (C, 2) mean, sd
    theta: np.ndarray                # (chains, draws, p)
    alpha: np.ndarray                # (chains, draws, C)
    K: np.ndarray                    # (chains, draws, C)
    sigma2: np.ndarray               # (chains, draws)
    y_mean: float
    y_sd: float
    med: np.ndarray                  # (C,) spend-per-capita scale
    n_base: int                      # columns before media block
    accept: np.ndarray               # (C,) MH acceptance after warmup
    base_names: list[str] = field(default_factory=list)

    # flat views (chains*draws)
    def flat(self, a):
        return a.reshape(-1, *a.shape[2:])


def _base_design(d: MMMData):
    """Geo intercepts + standardized controls + trend + 2 annual Fourier pairs."""
    G, T = d.revenue.shape
    cols, names = [], []
    for g in range(G):
        oh = np.zeros((G, T)); oh[g] = 1
        cols.append(oh.ravel()); names.append(f"intercept:{d.geos[g]}")
    t = np.arange(T) / 52.0
    doy = 2 * np.pi * d.weeks.dayofyear.to_numpy() / 365.25
    extra = [(f"control:{n}", d.controls[i]) for i, n in enumerate(d.control_names)]
    extra += [("trend", np.broadcast_to(t, (G, T)))]
    for k in (1, 2):
        extra += [(f"season:sin{k}", np.broadcast_to(np.sin(k * doy), (G, T))),
                  (f"season:cos{k}", np.broadcast_to(np.cos(k * doy), (G, T)))]
    for name, v in extra:
        v = np.asarray(v, float).ravel()
        sd = v.std()
        cols.append((v - v.mean()) / sd if sd > 0 else v * 0)
        names.append(name)
    return np.column_stack(cols), names, G


def _log_marginal(Xw, yw, m0, v0, sig2):
    """log N(yw | Xw m0, sig2 I + Xw diag(v0) Xw') via Woodbury.
    Returns (logp, cholesky of posterior precision, z) for reuse."""
    r = yw - Xw @ m0
    lam = Xw.T @ Xw / sig2
    lam[np.diag_indices_from(lam)] += 1 / v0
    Lc = np.linalg.cholesky(lam)
    z = np.linalg.solve(Lc, Xw.T @ r / sig2)
    logdet = len(yw) * np.log(sig2) + np.log(v0).sum() + 2 * np.log(np.diag(Lc)).sum()
    return -0.5 * (logdet + r @ r / sig2 - z @ z), Lc, z


def fit(d: MMMData, roi_prior, *, chains=4, warmup=800, draws=800, seed=7,
        rho_prior=(3.0, 0.18), K_prior_sd=0.7, verbose=True) -> Fit:
    """roi_prior: (C, 2) array of (mean, sd) on each channel's ROI."""
    rng = np.random.default_rng(seed)
    roi_prior = np.asarray(roi_prior, float)
    C, G, T = d.spend.shape
    n = G * T

    pc = d.revenue / d.pop[:, None]
    y_mean, y_sd = pc.mean(), pc.std()
    y = ((pc - y_mean) / y_sd).ravel()
    sw = np.sqrt(np.repeat(d.pop / d.pop.mean(), T))   # big geos = less noisy rows
    yw = y * sw

    xs = d.spend / d.pop[None, :, None]
    med = np.array([np.median(xs[c][xs[c] > 0]) for c in range(C)])
    xs = xs / med[:, None, None]
    S = d.spend.sum(axis=(1, 2))
    popw = d.pop[:, None]

    Zb, base_names, _ = _base_design(d)
    nb = Zb.shape[1]
    use_geo = G > 1
    p = nb + C + (G * C if use_geo else 0)
    onehot = np.repeat(np.eye(G), T, axis=0)            # (n, G)

    v0_base = np.r_[np.full(G, 9.0), np.full(nb - G, 1.0)]

    def media_cols(c, a, k):
        H = hill(adstock(xs[c], a), k)                  # (G, T)
        A = y_sd * (popw * H).sum() / S[c]              # ROI per unit beta_bar
        h = H.ravel()
        blk = [h[:, None]] + ([onehot * h[:, None]] if use_geo else [])
        return np.hstack(blk), A

    def assemble(cols, As, rho2):
        # theta layout: [base | beta_bar (C) | delta (C x G)] — keep X in that order
        X = np.hstack([Zb] + [m[:, :1] for m in cols] + [m[:, 1:] for m in cols])
        b0 = roi_prior[:, 0] / As                       # prior mean beta_bar
        m0 = np.r_[np.zeros(nb), b0, np.zeros(G * C if use_geo else 0)]
        v0 = np.r_[v0_base, (roi_prior[:, 1] / As) ** 2,
                   np.repeat(rho2 * b0 ** 2, G) if use_geo else []]
        return X * sw[:, None], m0, v0, b0

    def log_prior_shape(a, k):  # alpha ~ U(0,1) in logit space, log K ~ N(0, sd)
        return np.log(a) + np.log1p(-a) - 0.5 * (np.log(k) / K_prior_sd) ** 2

    out_theta = np.zeros((chains, draws, p))
    out_a = np.zeros((chains, draws, C)); out_k = np.zeros((chains, draws, C))
    out_s = np.zeros((chains, draws))
    acc_total = np.zeros(C)

    for ch in range(chains):
        a = rng.uniform(0.2, 0.7, C); k = np.exp(rng.normal(0, 0.3, C))
        sig2 = 0.1 * rng.uniform(0.5, 1.5); rho2 = np.full(C, 0.09)
        step = np.full(C, 0.3)
        built = [media_cols(c, a[c], k[c]) for c in range(C)]
        cols = [b[0] for b in built]; As = np.array([b[1] for b in built])
        acc = np.zeros(C)

        for it in range(warmup + draws):
            # 1) adstock/Hill shapes, linear part integrated out
            for c in range(C):
                Xw, m0, v0, _ = assemble(cols, As, rho2)
                cur, _, _ = _log_marginal(Xw, yw, m0, v0, sig2)
                cur += log_prior_shape(a[c], k[c])
                la = np.log(a[c] / (1 - a[c])) + step[c] * rng.normal()
                a_new = 1 / (1 + np.exp(-la))
                k_new = k[c] * np.exp(step[c] * rng.normal())
                a_new = min(max(a_new, 1e-4), 1 - 1e-4)
                col_new, A_new = media_cols(c, a_new, k_new)
                cols_p = cols.copy(); cols_p[c] = col_new
                As_p = As.copy(); As_p[c] = A_new
                Xw, m0, v0, _ = assemble(cols_p, As_p, rho2)
                new, _, _ = _log_marginal(Xw, yw, m0, v0, sig2)
                new += log_prior_shape(a_new, k_new)
                ok = np.log(rng.uniform()) < new - cur
                if ok:
                    a[c], k[c], cols, As = a_new, k_new, cols_p, As_p
                if it < warmup:   # Robbins-Monro step adaptation toward 30% acceptance
                    step[c] *= np.exp((ok - 0.3) / np.sqrt(it + 1))
                else:
                    acc[c] += ok

            # 2) all linear coefficients from their exact Gaussian conditional
            Xw, m0, v0, b0 = assemble(cols, As, rho2)
            _, Lc, z = _log_marginal(Xw, yw, m0, v0, sig2)
            theta = m0 + np.linalg.solve(Lc.T, z + rng.normal(size=p))

            # 3) noise + pooling scales (inverse-gamma conditionals)
            e = yw - Xw @ theta
            sig2 = 1 / rng.gamma(1.0 + n / 2, 1 / (0.01 + e @ e / 2))
            if use_geo:
                delta = theta[nb + C:].reshape(C, G) / b0[:, None]
                rho2 = 1 / rng.gamma(rho_prior[0] + G / 2,
                                     1 / (rho_prior[1] + (delta ** 2).sum(1) / 2))

            if it >= warmup:
                j = it - warmup
                out_theta[ch, j], out_a[ch, j], out_k[ch, j], out_s[ch, j] = theta, a, k, sig2
        acc_total += acc / draws
        if verbose:
            print(f"  chain {ch + 1}/{chains} done · MH acceptance "
                  + ", ".join(f"{n_}:{r:.0%}" for n_, r in zip(d.channels, acc / draws)))

    return Fit(d, roi_prior, out_theta, out_a, out_k, out_s, y_mean, y_sd, med,
               nb, acc_total / chains, base_names)


# ---------------------------------------------------------------- posterior
def _betas(f: Fit, theta):
    C, G = len(f.data.channels), len(f.data.geos)
    bbar = theta[f.n_base:f.n_base + C]
    if G > 1:
        return bbar[:, None] + theta[f.n_base + C:].reshape(C, G)
    return bbar[:, None]


def media_revenue(f: Fit, theta, alpha, K, mult=None) -> np.ndarray:
    """(C, G, T) revenue attributable to each channel at spend x mult."""
    d = f.data
    C = len(d.channels)
    mult = np.ones(C) if mult is None else np.asarray(mult, float)
    xs = d.spend / d.pop[None, :, None] / f.med[:, None, None]
    B = _betas(f, theta)
    out = np.empty(d.spend.shape)
    for c in range(C):
        H = hill(adstock(xs[c] * mult[c], alpha[c]), K[c])
        out[c] = d.pop[:, None] * f.y_sd * B[c][:, None] * H
    return out


def predict_revenue(f: Fit, theta, alpha, K, mult=None) -> np.ndarray:
    """(G, T) predicted revenue — baseline (intercepts, controls, season) + media."""
    d = f.data
    Zb, _, _ = _base_design(d)
    G, T = d.revenue.shape
    base = (Zb @ theta[:f.n_base]).reshape(G, T)
    base = d.pop[:, None] * (f.y_mean + f.y_sd * base)
    return base + media_revenue(f, theta, alpha, K, mult).sum(0)


def response_curve(f: Fit, theta, alpha, K, c: int, mults: np.ndarray) -> np.ndarray:
    """Incremental revenue of channel c if its whole spend plan is scaled by
    each multiplier (adstock is linear, so only the Hill step is recomputed)."""
    d = f.data
    xs = d.spend[c] / d.pop[:, None] / f.med[c]
    ad = adstock(xs, alpha[c])
    w = d.pop[:, None] * f.y_sd * _betas(f, theta)[c][:, None]
    return np.array([(w * hill(m * ad, K[c])).sum() for m in mults])


def counterfactual_roi(f: Fit, theta, alpha, K) -> np.ndarray:
    """ROI_c = [revenue(actual) - revenue(channel c zeroed)] / spend_c."""
    C = len(f.data.channels)
    actual = predict_revenue(f, theta, alpha, K).sum()
    roi = np.empty(C)
    for c in range(C):
        m = np.ones(C); m[c] = 0
        roi[c] = (actual - predict_revenue(f, theta, alpha, K, m).sum()) / f.data.spend[c].sum()
    return roi


def rhat(x: np.ndarray) -> float:
    """Split-R-hat for one scalar, x shape (chains, draws)."""
    h = x.shape[1] // 2
    s = np.vstack([x[:, :h], x[:, h:2 * h]])
    m, n = s.shape
    W = s.var(axis=1, ddof=1).mean()
    B = n * s.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W)) if W > 0 else 1.0


def optimize_budget(curves_mean: np.ndarray, mults: np.ndarray, spend: np.ndarray,
                    bounds: float = 0.3, step: float = 0.005):
    """Greedy marginal allocation on posterior-mean response curves with the
    total budget held fixed and each channel kept within +/- bounds.
    Greedy is optimal here because Hill curves are concave.
    curves_mean: (C, M) incremental revenue at spend x mults."""
    C = len(spend)
    total = spend.sum()
    lo, hi = spend * (1 - bounds), spend * (1 + bounds)
    alloc = lo.copy()
    rev = lambda c, s: np.interp(s / spend[c], mults, curves_mean[c])
    inc = step * total
    while alloc.sum() < total - 1e-9:
        room = np.minimum(hi - alloc, total - alloc.sum())
        gains = np.array([rev(c, alloc[c] + min(inc, room[c])) - rev(c, alloc[c])
                          if room[c] > 1e-9 else -np.inf for c in range(C)])
        best = int(np.argmax(gains))
        alloc[best] += min(inc, room[best])
    return alloc
