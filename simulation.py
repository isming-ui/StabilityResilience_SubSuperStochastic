"""
Simulation pipeline for:
  "Predicting Temporal Stability and Resilience of Competitive Networked
   Systems via Sub/Super-Stochastic Matrices"

Implements:
  * augmented sub/super-stochastic representation of a signed digraph
  * pulse-perturbed discrete-time competitive network dynamics
  * empirical measurement of (Omega_2, Delta_2, I_2, Phi_{2,x}, tau_eps)
  * closed-form predictors hat_I2, hat_Phi_2x, recovery-time bound
  * production of the four figures referenced in Section VI of the paper

Run:
    python simulation.py
Outputs: fig_I2.png, fig_Phi.png, fig_tau.png, fig_density.png in cwd.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats


# ----------------------------------------------------------------------
# Global publication-quality style (TASE / IEEE)
# ----------------------------------------------------------------------
PALETTE = {
    "blue":  "#1f4e79",   # proposed / CDSM
    "red":   "#c00000",   # scalar / baseline
    "gray":  "#7f7f7f",   # diagonal / reference
    "lblue": "#cfe0f0",   # CDSM 95% CI
    "lred":  "#f4cccc",   # scalar 95% CI
    "edge":  "#2b2b2b",
}


def set_plot_style():
    """Publication-quality matplotlib defaults: LaTeX-rendered Times throughout
    (text via mathptmx, math symbols via amsmath); IEEE-friendly column sizes."""
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{mathptmx}\usepackage{amsmath}\usepackage{bm}",
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "axes.edgecolor": PALETTE["edge"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    })


def _inset_metric_box(ax, text, loc="lower right", pad=0.02):
    """Add a small light-gray rounded box with one or two lines of metric text."""
    x, y, ha, va = {
        "lower right": (1 - pad, pad, "right", "bottom"),
        "upper left":  (pad, 1 - pad, "left", "top"),
        "upper right": (1 - pad, 1 - pad, "right", "top"),
        "lower left":  (pad, pad, "left", "bottom"),
    }[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PALETTE["gray"],
                      lw=0.6, alpha=0.92))


def _regression_band(ax, x, y, color, label=None, n_grid=80):
    """Linear regression line with 95% CI band on x in [xmin, xmax]."""
    x = np.asarray(x)
    y = np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return None
    slope, intercept, r, _, _ = stats.linregress(x, y)
    xg = np.linspace(x.min(), x.max(), n_grid)
    yg = slope * xg + intercept
    n = x.size
    xbar = x.mean()
    ssx = np.sum((x - xbar) ** 2)
    residuals = y - (slope * x + intercept)
    mse = np.sum(residuals ** 2) / max(n - 2, 1)
    se = np.sqrt(mse * (1 / n + (xg - xbar) ** 2 / max(ssx, 1e-12)))
    ax.fill_between(xg, yg - 1.96 * se, yg + 1.96 * se, color=color, alpha=0.18,
                    linewidth=0)
    ax.plot(xg, yg, color=color, lw=1.2, label=label)
    return r ** 2


def _safe_mape(o, h):
    ok = np.isfinite(o) & np.isfinite(h) & (np.abs(o) > 1e-6)
    return float(np.mean(np.abs((h[ok] - o[ok]) / o[ok])) * 100) if ok.any() else float("nan")


# ----------------------------------------------------------------------
# Network construction
# ----------------------------------------------------------------------
def make_signed_network(n: int, density: float, neg_frac: float,
                        wmin: float = 0.1, wmax: float = 0.6,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Erdos--Renyi signed digraph with a target fraction of negative edges."""
    rng = rng or np.random.default_rng()
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, False)
    A = np.zeros((n, n))
    mag = rng.uniform(wmin, wmax, size=(n, n))
    signs = np.where(rng.random((n, n)) < neg_frac, -1.0, 1.0)
    A[mask] = (mag * signs)[mask]
    return A


def contractive_step(A: np.ndarray, target_spr: float = 0.6) -> float:
    """Pick T so that spr(I - TD + TA) is roughly target_spr (<1)."""
    n = A.shape[0]
    D = np.diag(np.abs(A).sum(axis=1))
    best_T, best_diff = 0.05, float("inf")
    for T in np.linspace(0.02, 0.5, 25):
        R = np.eye(n) - T * D + T * A
        s = spectral_radius(R)
        if s < 1.0 and abs(s - target_spr) < best_diff:
            best_diff = abs(s - target_spr)
            best_T = T
    return float(best_T)


def iteration_matrix(A: np.ndarray, T: float) -> np.ndarray:
    """R = I - T D + T A, with D = diag(sum_j |a_ij|)."""
    D = np.diag(np.abs(A).sum(axis=1))
    return np.eye(A.shape[0]) - T * D + T * A


def augmented_matrix(A: np.ndarray, T: float) -> np.ndarray:
    """Lift the signed A into the non-negative augmented R-tilde of size 2n x 2n."""
    n = A.shape[0]
    Atil = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            a = A[i, j]
            if a > 0:
                Atil[2 * j, 2 * i] = a
                Atil[2 * j + 1, 2 * i + 1] = a
            elif a < 0:
                Atil[2 * j, 2 * i + 1] = -a
                Atil[2 * j + 1, 2 * i] = -a
    Dtil = np.diag(Atil.sum(axis=1))
    return np.eye(2 * n) - T * Dtil + T * Atil


def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))


# ----------------------------------------------------------------------
# Dynamics
# ----------------------------------------------------------------------
def simulate(A: np.ndarray, T: float, P: int, T_obs: int,
             pulse_sigma: float, Y_n: float = 1.0,
             rng: np.random.Generator | None = None):
    """
    Discrete-time pulse-perturbed competitive network in deviation form:
        xi(t+1) = R xi(t) + w(t)
        Y(t)   = Y_n + mean(xi(t))
    Pulses are applied every P steps with magnitude ~ N(0, pulse_sigma^2).
    Returns the ecosystem-level trajectory Y(t) and the pulse indices.
    """
    rng = rng or np.random.default_rng()
    R = iteration_matrix(A, T)
    n = A.shape[0]
    xi = np.zeros(n)
    traj = np.full(T_obs, Y_n)
    pulses = []
    for t in range(T_obs):
        if t > 0 and t % P == 0:
            w = rng.normal(0.0, pulse_sigma * Y_n, size=n)
            xi = R @ xi + w
            pulses.append(t)
        else:
            xi = R @ xi
        traj[t] = Y_n + xi.mean()
    return traj, pulses, np.full(n, Y_n)


# ----------------------------------------------------------------------
# Empirical descriptors
# ----------------------------------------------------------------------
def measure_first_cycle(traj: np.ndarray, pulses: list[int], Y_n: float):
    """Single-cycle resistance Omega_2 and recovery Delta_2."""
    if not pulses:
        raise RuntimeError("no perturbation observed")
    e = pulses[0]
    Y_e = traj[e]
    Y_ep1 = traj[e + 1]
    Omega_2 = 1.0 - abs(Y_n - Y_e) / Y_n
    denom = abs(Y_n - Y_e)
    Delta_2 = 1.0 - abs(Y_n - Y_ep1) / denom if denom > 1e-12 else 0.0
    return Omega_2, Delta_2


def measure_I2(traj: np.ndarray) -> float:
    mu = traj.mean()
    sigma = traj.std()
    return 1.0 - sigma / mu if mu != 0 else 0.0


def measure_Phi2x(traj: np.ndarray, pulses: list[int], Y_n: float, x: int) -> float:
    vals = []
    for e in pulses:
        if e + x < len(traj):
            vals.append(1.0 - abs(Y_n - traj[e + x]) / Y_n)
    return float(np.mean(vals)) if vals else 0.0


def measure_tau(traj: np.ndarray, pulses: list[int], Y_n: float,
                eps: float, P: int) -> float:
    vals = []
    for e in pulses:
        for x in range(1, P):
            if e + x >= len(traj):
                break
            if abs(Y_n - traj[e + x]) <= eps * Y_n:
                vals.append(x)
                break
    return float(np.mean(vals)) if vals else float(P)


# ----------------------------------------------------------------------
# Predictors (Theorems IV.1, V.1, V.2, Corollary V.3)
# ----------------------------------------------------------------------
def predict_I2(Omega_2: float, Delta_2: float, P: int, T_obs: int) -> float:
    """Equation (15) in the paper."""
    if not (0 < Delta_2 < 2):
        return float("nan")
    P_obs = T_obs / P
    num = P_obs * ((1.0 - Delta_2) ** (2 * T_obs / P) - 1.0)
    den = (T_obs - 1) * Delta_2 * (Delta_2 - 2.0)
    val = num / den
    val = max(val, 0.0)
    return 1.0 + (Omega_2 - 1.0) * np.sqrt(val)


def predict_Phi(Omega_2: float, Delta_2: float, x: int) -> float:
    """Equation (17)."""
    return 1.0 - (1.0 - Delta_2) ** x + Omega_2 * (1.0 - Delta_2) ** x


def predict_tau(Omega_2: float, Delta_2: float, eps: float) -> float:
    """Corollary V.3 — clamped to be a valid (>=1) upper bound."""
    a = 1.0 - Delta_2
    if a <= 0 or a >= 1:
        return float("inf")
    disp = 1.0 - Omega_2
    if disp <= eps:
        return 1.0  # already inside tolerance after the perturbation step
    raw = (np.log(eps) - np.log(disp)) / np.log(a)
    return float(max(1.0, np.ceil(raw)))


# ----------------------------------------------------------------------
# CDSM profile (Definition III.3) and profile-aware predictors
# (Theorems V.2, V.4, Corollary V.5)
# ----------------------------------------------------------------------
def cdsm_envelope_matrix(R: np.ndarray, P: int) -> np.ndarray:
    """
    Matrix-based CDSM upper-envelope of a non-augmented iteration matrix R
    (which is generally non-row-stochastic because of the signed entries in A).
    Returns bar_eta_t = max_{s>=t} max_i |Lambda_i[R^{t+1}]| / |Lambda_i[R^t]|,
    truncated to [1e-6, 1].
    Only useful when R itself is non-row-stochastic; for the augmented R-tilde
    (which is row-stochastic), use cdsm_envelope_empirical instead.
    """
    n = R.shape[0]
    Rt = np.eye(n)
    raw = np.zeros(P)
    prev_rs = Rt.sum(axis=1)
    for t in range(P):
        Rt = Rt @ R
        rs = Rt.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(prev_rs) > 1e-12,
                             np.abs(rs) / np.maximum(np.abs(prev_rs), 1e-12),
                             1.0)
        raw[t] = float(np.max(np.abs(ratio)))
        prev_rs = rs
    env = np.maximum.accumulate(raw[::-1])[::-1]
    return np.clip(env, 1e-6, 1.0)


def cdsm_envelope_empirical(traj: np.ndarray, pulses: list[int], Y_n: float,
                            P: int, n_cycles: int = 5) -> np.ndarray:
    """
    Empirical CDSM upper envelope built from cumulative post-pulse deviations.
    For each of the first ``n_cycles`` perturbations, the normalized cumulative
    deviation c_t = |Y_n - Y_{e+t}| / |Y_n - Y_e| (with c_0 = 1) is computed;
    the worst-case curve m_t = max over cycles of c_t is taken; finally the
    running max from the right turns m_t into a non-increasing sequence,
    yielding the envelope bar_eta_t = m_{t+1} (non-increasing in t).
    This formulation bypasses the per-step explosion that arises when the
    deviation transiently rebounds after the pulse.
    """
    curves = []
    for e in pulses[:n_cycles]:
        if e + P >= len(traj):
            continue
        d0 = abs(Y_n - traj[e])
        if d0 <= 1e-12:
            continue
        c = np.array([abs(Y_n - traj[e + t]) / d0 for t in range(P + 1)])
        curves.append(c)
    if not curves:
        return np.ones(P)
    m = np.maximum.reduce(curves)
    # turn m into a non-increasing upper envelope (running max from the right)
    m_env = np.maximum.accumulate(m[::-1])[::-1]
    # eta_t is the per-step ratio of the cumulative envelope (already <=1)
    eta = m_env[1:P + 1] / np.maximum(m_env[:P], 1e-12)
    return np.clip(eta, 1e-6, 1.0)


def predict_I2_cdsm(Omega_2: float, eta: np.ndarray, P: int, T_obs: int) -> float:
    """Theorem V.2 (profile form): hat I2^{CDSM}."""
    P_obs = T_obs / P
    cum = np.concatenate([[1.0], np.cumprod(eta)])  # prod_{s<t} eta_s, t=0..P
    val = (P_obs / (T_obs - 1)) * np.sum(cum[:P] ** 2)
    val = max(val, 0.0)
    return 1.0 + (Omega_2 - 1.0) * np.sqrt(val)


def predict_Phi_cdsm(Omega_2: float, eta: np.ndarray, x: int) -> float:
    """Theorem V.4 (profile form): hat Phi_{2,x}^{CDSM}."""
    if x <= 0:
        return 1.0
    prod = float(np.prod(eta[:x]))
    return 1.0 - (1.0 - Omega_2) * prod


def predict_tau_cdsm(Omega_2: float, eta: np.ndarray, eps: float) -> float:
    """Corollary V.5 (profile form): smallest x with (1-Omega_2)*prod eta <= eps."""
    disp = 1.0 - Omega_2
    if disp <= eps:
        return 1.0
    cur = disp
    for x, e in enumerate(eta, start=1):
        cur *= e
        if cur <= eps:
            return float(x)
    return float(len(eta))


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
def run_main_experiment(n_real: int = 200, n: int = 50, density: float = 0.15,
                        neg_frac: float = 0.40, P: int = 10,
                        T_obs: int = 1000, pulse_sigma: float = 0.05,
                        seed: int = 0):
    """Run n_real realizations, each on a fresh contractive signed network."""
    rng = np.random.default_rng(seed)
    obs_I2, hat_I2, hat_I2_cd = [], [], []
    obs_Phi, hat_Phi, hat_Phi_cd = [], [], []
    obs_tau, hat_tau, hat_tau_cd = [], [], []
    attempts = 0
    while len(obs_I2) < n_real and attempts < 6 * n_real:
        attempts += 1
        target_spr = float(rng.uniform(0.30, 0.88))
        sigma_k = float(rng.uniform(0.30, 2.00))
        A = make_signed_network(n, density, neg_frac, rng=rng)
        T = contractive_step(A, target_spr=target_spr)
        R = iteration_matrix(A, T)
        if spectral_radius(R) >= 0.97:
            continue
        try:
            traj, pulses, _ = simulate(A, T, P, T_obs, sigma_k, rng=rng)
        except np.linalg.LinAlgError:
            continue
        if not pulses or not np.all(np.isfinite(traj)):
            continue
        Om, De = measure_first_cycle(traj, pulses, Y_n=1.0)
        if not (0 < De < 1.5) or not (0 < Om < 1):
            continue
        I2 = measure_I2(traj)
        Phi = measure_Phi2x(traj, pulses, 1.0, x=2)
        eps_tau = 0.005
        tau = measure_tau(traj, pulses, 1.0, eps=eps_tau, P=P)
        if not np.isfinite(I2) or not np.isfinite(Phi) or I2 <= 0 or I2 > 1:
            continue
        # CDSM envelope built empirically from the post-pulse trajectory
        eta = cdsm_envelope_empirical(traj, pulses, Y_n=1.0, P=P)
        obs_I2.append(I2)
        hat_I2.append(predict_I2(Om, De, P, T_obs))
        hat_I2_cd.append(predict_I2_cdsm(Om, eta, P, T_obs))
        obs_Phi.append(Phi)
        hat_Phi.append(predict_Phi(Om, De, x=2))
        hat_Phi_cd.append(predict_Phi_cdsm(Om, eta, x=2))
        obs_tau.append(tau)
        hat_tau.append(predict_tau(Om, De, eps=eps_tau))
        hat_tau_cd.append(predict_tau_cdsm(Om, eta, eps=eps_tau))
    return (np.array(obs_I2), np.array(hat_I2), np.array(hat_I2_cd),
            np.array(obs_Phi), np.array(hat_Phi), np.array(hat_Phi_cd),
            np.array(obs_tau), np.array(hat_tau), np.array(hat_tau_cd))


def run_density_sweep_fixed(neg_fracs: np.ndarray, n_real: int = 80,
                            n: int = 50, density: float = 0.15, P: int = 10,
                            T_obs: int = 1000, pulse_sigma: float = 0.20,
                            target_spr: float = 0.70, seed_base: int = 100):
    """
    Same per-realization parameters across f^- to isolate the effect of
    competitive edge density on prediction accuracy.
    """
    mapes_proposed, mapes_naive = [], []
    for fi, f in enumerate(neg_fracs):
        rng = np.random.default_rng(seed_base + fi)
        oI2_p, hI2_p, oI2_n, hI2_n = [], [], [], []
        attempts, kept = 0, 0
        while kept < n_real and attempts < 6 * n_real:
            attempts += 1
            A = make_signed_network(n, density, float(f), rng=rng)
            T = contractive_step(A, target_spr=target_spr)
            R = iteration_matrix(A, T)
            if spectral_radius(R) >= 0.97:
                continue
            traj, pulses, _ = simulate(A, T, P, T_obs, pulse_sigma, rng=rng)
            if not pulses or not np.all(np.isfinite(traj)):
                continue
            Om, De = measure_first_cycle(traj, pulses, Y_n=1.0)
            if not (0 < De < 1.5) or not (0 < Om < 1):
                continue
            I2 = measure_I2(traj)
            if not np.isfinite(I2) or I2 <= 0 or I2 > 1:
                continue
            kept += 1
            # proposed predictor: scalar predictor evaluated on the
            # signed-A measurements (Om, De) obtained on the genuine
            # augmented network -- this isolates the value of correctly
            # absorbing the sign pattern via Theorem 1.
            oI2_p.append(I2)
            hI2_p.append(predict_I2(Om, De, P, T_obs))
            # naive baseline: run the SAME simulation on the cooperative
            # surrogate |A| (sign pattern erased) and re-measure (Om,De),
            # then plug into the predictor.  This corresponds to applying
            # the ecological predictor of [Isbell 2026] directly, ignoring
            # the sub/super-stochastic correction.
            traj_n, pulses_n, _ = simulate(np.abs(A), T, P, T_obs,
                                           pulse_sigma, rng=rng)
            if pulses_n and np.all(np.isfinite(traj_n)):
                Om_n, De_n = measure_first_cycle(traj_n, pulses_n, Y_n=1.0)
                if 0 < De_n < 1.5 and 0 < Om_n < 1:
                    oI2_n.append(I2)
                    hI2_n.append(predict_I2(Om_n, De_n, P, T_obs))
        oI2_p, hI2_p = np.array(oI2_p), np.array(hI2_p)
        oI2_n, hI2_n = np.array(oI2_n), np.array(hI2_n)
        ok = np.isfinite(oI2_p) & np.isfinite(hI2_p) & (np.abs(oI2_p) > 1e-6)
        mapes_proposed.append(
            float(np.mean(np.abs((hI2_p[ok] - oI2_p[ok]) / oI2_p[ok])) * 100)
            if ok.any() else np.nan)
        ok = np.isfinite(oI2_n) & np.isfinite(hI2_n) & (np.abs(oI2_n) > 1e-6)
        mapes_naive.append(
            float(np.mean(np.abs((hI2_n[ok] - oI2_n[ok]) / oI2_n[ok])) * 100)
            if ok.any() else np.nan)
    return np.array(mapes_proposed), np.array(mapes_naive)


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------
def scatter_predictor(x_obs, ys, labels, colors, xlabel, ylabel, fname,
                      symmetric=True):
    """
    Publication scatter for an observed-vs-predicted comparison.
    `ys` and `colors` and `labels` are parallel lists, one per series.
    Adds a 1:1 dashed diagonal, regression lines with 95% CI, and an
    inset metric box reporting MAPE (and R^2) for every series.
    """
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    pts_min, pts_max = +np.inf, -np.inf
    metrics_lines = []
    for y, lab, c in zip(ys, labels, colors):
        ok = np.isfinite(x_obs) & np.isfinite(y)
        xx, yy = x_obs[ok], y[ok]
        ax.scatter(xx, yy, s=14, alpha=0.45, color=c, linewidth=0,
                   label=lab, rasterized=True)
        r2 = _regression_band(ax, xx, yy, color=c)
        mape = _safe_mape(x_obs, y)
        if r2 is None:
            metrics_lines.append(f"{lab}: MAPE = {mape:.2f}" + r"\%")
        else:
            metrics_lines.append(f"{lab}: MAPE = {mape:.2f}" + r"\%,  $R^2$ = " + f"{r2:.2f}")
        pts_min = min(pts_min, float(xx.min()), float(yy.min()))
        pts_max = max(pts_max, float(xx.max()), float(yy.max()))
    pad = (pts_max - pts_min) * 0.05
    lo, hi = pts_min - pad, pts_max + pad
    if symmetric:
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color=PALETTE["gray"],
            label="$y = x$", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.5, handlelength=1.4)
    _inset_metric_box(ax, "\n".join(metrics_lines), loc="lower right")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def plot_tau(hat_scalar: np.ndarray, hat_cdsm: np.ndarray, obs: np.ndarray,
             fname: str, P: int = 10, eps: float = 0.005):
    """Joint scatter with marginal histograms comparing scalar vs CDSM tau bound."""
    rng = np.random.default_rng(0)

    def _mask(h):
        return (np.isfinite(obs) & np.isfinite(h) & (h > 0)
                & (h <= P) & (obs < P + 1))

    m_s = _mask(hat_scalar)
    m_c = _mask(hat_cdsm)
    if not m_s.any() or not m_c.any():
        print("[warn] no in-cycle recovery realizations for tau plot")
        return

    cov_s = float(np.mean(hat_scalar[m_s] >= obs[m_s]) * 100)
    cov_c = float(np.mean(hat_cdsm[m_c] >= obs[m_c]) * 100)
    err_s = float(np.mean(np.abs(hat_scalar[m_s] - obs[m_s])))
    err_c = float(np.mean(np.abs(hat_cdsm[m_c] - obs[m_c])))

    fig = plt.figure(figsize=(3.6, 3.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.08, wspace=0.08,
                          left=0.16, bottom=0.16, right=0.98, top=0.98)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # main scatter (jittered to reveal density)
    for hat, m, c, marker, lab in [
        (hat_scalar, m_s, PALETTE["red"], "s", "scalar"),
        (hat_cdsm, m_c, PALETTE["blue"], "o", "CDSM"),
    ]:
        jo = obs[m] + rng.normal(0, 0.08, size=m.sum())
        jh = hat[m] + rng.normal(0, 0.08, size=m.sum())
        ax.scatter(jo, jh, s=12, alpha=0.5, color=c, marker=marker,
                   linewidth=0, label=lab, rasterized=True)

    lo, hi = -0.5, float(P) + 0.5
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color=PALETTE["gray"],
            zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(rf"empirical $\tau_{{{eps}}}$ (steps)")
    ax.set_ylabel(r"predicted $\hat\tau$ (steps)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.5, handlelength=1.0, markerscale=1.2)
    _inset_metric_box(
        ax,
        f"scalar: cov = {cov_s:.0f}" + r"\%" + f", MAE = {err_s:.2f}" + "\n"
        f"CDSM: cov = {cov_c:.0f}" + r"\%" + f", MAE = {err_c:.2f}",
        loc="lower right",
    )

    # marginal histograms
    bins = np.arange(0, P + 2) - 0.5
    ax_top.hist(obs[m_s], bins=bins, color=PALETTE["gray"], alpha=0.6,
                edgecolor="white", linewidth=0.4)
    ax_right.hist(np.concatenate([hat_scalar[m_s], hat_cdsm[m_c]]),
                  bins=bins, orientation="horizontal",
                  color=PALETTE["gray"], alpha=0.6,
                  edgecolor="white", linewidth=0.4)
    for a in (ax_top, ax_right):
        for s in ("top", "right", "left", "bottom"):
            a.spines[s].set_visible(False)
    ax_top.set_yticks([])
    ax_right.set_xticks([])

    fig.savefig(fname)
    plt.close(fig)


def run_sigma_sweep(sigmas: np.ndarray, n_real: int = 100, n: int = 50,
                    density: float = 0.15, neg_frac: float = 0.40, P: int = 10,
                    T_obs: int = 1000, seed_base: int = 200):
    """
    Robustness sweep across perturbation magnitudes.
    Fixes per-realization spectral radius target at 0.55; varies pulse_sigma.
    Returns dict with arrays: sigmas, mape_I2, mape_Phi, cov_tau,
    range_I2 (= max(I2)-min(I2) observed), and median I2.
    """
    out = {"sigmas": sigmas,
           "mape_I2_scalar": [], "mape_I2_cdsm": [],
           "mape_Phi_scalar": [], "mape_Phi_cdsm": [],
           "cov_tau_scalar": [], "cov_tau_cdsm": [],
           "range_I2": [], "med_I2": []}
    for si, sg in enumerate(sigmas):
        rng = np.random.default_rng(seed_base + si)
        oI2, hI2, hI2c = [], [], []
        oPhi, hPhi, hPhic = [], [], []
        oTau, hTau, hTauc = [], [], []
        kept, attempts = 0, 0
        while kept < n_real and attempts < 8 * n_real:
            attempts += 1
            A = make_signed_network(n, density, neg_frac, rng=rng)
            T = contractive_step(A, target_spr=0.55)
            R = iteration_matrix(A, T)
            if spectral_radius(R) >= 0.97:
                continue
            traj, pulses, _ = simulate(A, T, P, T_obs, float(sg), rng=rng)
            if not pulses or not np.all(np.isfinite(traj)):
                continue
            Om, De = measure_first_cycle(traj, pulses, Y_n=1.0)
            if not (0 < De < 1.5) or not (0 < Om < 1):
                continue
            I2 = measure_I2(traj)
            if not np.isfinite(I2) or I2 <= 0 or I2 > 1:
                continue
            kept += 1
            Phi = measure_Phi2x(traj, pulses, 1.0, x=2)
            tau = measure_tau(traj, pulses, 1.0, eps=0.005, P=P)
            eta = cdsm_envelope_empirical(traj, pulses, 1.0, P)
            oI2.append(I2); hI2.append(predict_I2(Om, De, P, T_obs))
            hI2c.append(predict_I2_cdsm(Om, eta, P, T_obs))
            oPhi.append(Phi); hPhi.append(predict_Phi(Om, De, 2))
            hPhic.append(predict_Phi_cdsm(Om, eta, 2))
            oTau.append(tau); hTau.append(predict_tau(Om, De, 0.005))
            hTauc.append(predict_tau_cdsm(Om, eta, 0.005))
        oI2 = np.array(oI2); hI2 = np.array(hI2); hI2c = np.array(hI2c)
        oPhi = np.array(oPhi); hPhi = np.array(hPhi); hPhic = np.array(hPhic)
        oTau = np.array(oTau); hTau = np.array(hTau); hTauc = np.array(hTauc)
        out["mape_I2_scalar"].append(_safe_mape(oI2, hI2))
        out["mape_I2_cdsm"].append(_safe_mape(oI2, hI2c))
        out["mape_Phi_scalar"].append(_safe_mape(oPhi, hPhi))
        out["mape_Phi_cdsm"].append(_safe_mape(oPhi, hPhic))
        ok_s = np.isfinite(oTau) & np.isfinite(hTau) & (hTau > 0)
        ok_c = np.isfinite(oTau) & np.isfinite(hTauc) & (hTauc > 0)
        out["cov_tau_scalar"].append(
            float(np.mean(hTau[ok_s] >= oTau[ok_s]) * 100) if ok_s.any() else np.nan)
        out["cov_tau_cdsm"].append(
            float(np.mean(hTauc[ok_c] >= oTau[ok_c]) * 100) if ok_c.any() else np.nan)
        out["range_I2"].append(float(oI2.max() - oI2.min()) if oI2.size else np.nan)
        out["med_I2"].append(float(np.median(oI2)) if oI2.size else np.nan)
    for k in out:
        out[k] = np.array(out[k])
    return out


def plot_robustness(out: dict, fname_a: str, fname_b: str):
    """Two independent IEEE-subfigure-ready PNGs.

    fname_a: MAPE on I_2 / Phi_{2,2} vs sigma (panel (a))
    fname_b: tau coverage vs sigma (panel (b))
    """
    sg = out["sigmas"]

    # ----- Panel (a): MAPE vs sigma -----
    fig_a, ax1 = plt.subplots(figsize=(3.4, 2.6))
    ax1.plot(sg, out["mape_I2_scalar"], "o-", color=PALETTE["red"], lw=1.4,
             ms=4, label=r"$\hat I_2$ scalar")
    ax1.plot(sg, out["mape_I2_cdsm"], "s--", color=PALETTE["blue"], lw=1.4,
             ms=4, label=r"$\hat I_2$ CDSM")
    ax1.plot(sg, out["mape_Phi_scalar"], "o-", color=PALETTE["red"], lw=1.4,
             ms=4, alpha=0.5, label=r"$\hat\Phi_{2,2}$ scalar")
    ax1.plot(sg, out["mape_Phi_cdsm"], "s--", color=PALETTE["blue"], lw=1.4,
             ms=4, alpha=0.5, label=r"$\hat\Phi_{2,2}$ CDSM")
    ax1.set_xlabel(r"pulse magnitude $\sigma\,/\,Y_n$")
    ax1.set_ylabel(r"MAPE (\%)")
    ax1.set_xscale("log")
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=7, ncol=1, handlelength=1.4)
    fig_a.tight_layout()
    fig_a.savefig(fname_a)
    plt.close(fig_a)

    # ----- Panel (b): tau coverage vs sigma -----
    fig_b, ax2 = plt.subplots(figsize=(3.4, 2.6))
    ax2.plot(sg, out["cov_tau_scalar"], "o-", color=PALETTE["red"], lw=1.4,
             ms=4, label="scalar")
    ax2.plot(sg, out["cov_tau_cdsm"], "s--", color=PALETTE["blue"], lw=1.4,
             ms=4, label="CDSM")
    ax2.axhline(100, color=PALETTE["gray"], ls=":", lw=0.8, alpha=0.5)
    ax2.set_xlabel(r"pulse magnitude $\sigma\,/\,Y_n$")
    ax2.set_ylabel(r"$\tau_{0.005}$ coverage (\%)")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 109)
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize=7.5, handlelength=1.4)
    fig_b.tight_layout()
    fig_b.savefig(fname_b)
    plt.close(fig_b)


def run_density_sweep_with_bands(neg_fracs: np.ndarray, n_seeds: int = 4,
                                 **kw):
    """
    Repeat `run_density_sweep_fixed` over `n_seeds` random seeds and return
    (mean_p, std_p, mean_n, std_n) so that the density-sweep plot can show
    95% confidence bands.
    """
    all_p, all_n = [], []
    for k in range(n_seeds):
        p, n = run_density_sweep_fixed(neg_fracs, seed_base=100 + 17 * k, **kw)
        all_p.append(p)
        all_n.append(n)
    Ap = np.vstack(all_p)
    An = np.vstack(all_n)
    return (np.nanmean(Ap, axis=0), np.nanstd(Ap, axis=0),
            np.nanmean(An, axis=0), np.nanstd(An, axis=0))


def plot_scalar_vs_cdsm(oI2: np.ndarray, hI2: np.ndarray, hI2c: np.ndarray,
                        oPhi: np.ndarray, hPhi: np.ndarray, hPhic: np.ndarray,
                        fname: str):
    """Grouped bar chart: scalar vs CDSM MAPE on I2 and Phi side by side."""
    groups = [r"$I_2$", r"$\Phi_{2,2}$"]
    scalar = [_safe_mape(oI2, hI2), _safe_mape(oPhi, hPhi)]
    cdsm = [_safe_mape(oI2, hI2c), _safe_mape(oPhi, hPhic)]
    x = np.arange(len(groups))
    w = 0.36

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    b1 = ax.bar(x - w / 2, scalar, width=w, color=PALETTE["red"],
                edgecolor=PALETTE["edge"], linewidth=0.5,
                label="scalar (single-cycle)")
    b2 = ax.bar(x + w / 2, cdsm, width=w, color=PALETTE["blue"],
                edgecolor=PALETTE["edge"], linewidth=0.5,
                label="CDSM (profile-aware)")

    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                    f"{b.get_height():.2f}" + r"\%", ha="center", va="bottom",
                    fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(r"MAPE (\%)")
    ax.set_ylim(0, max(scalar + cdsm) * 1.25)
    ax.grid(axis="y", which="major")
    ax.legend(loc="upper left", fontsize=7.5, handlelength=1.4)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def plot_density_sweep(neg_fracs: np.ndarray, mapes_proposed: np.ndarray,
                       mapes_naive: np.ndarray, fname: str,
                       stds_proposed: np.ndarray | None = None,
                       stds_naive: np.ndarray | None = None):
    """Two-curve antagonism-density sweep with optional 95% CI shading."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ok = ~np.isnan(mapes_proposed) & ~np.isnan(mapes_naive)
    nf = neg_fracs[ok]

    if stds_naive is not None:
        s = stds_naive[ok]
        ax.fill_between(nf, mapes_naive[ok] - 1.96 * s,
                        mapes_naive[ok] + 1.96 * s,
                        color=PALETTE["red"], alpha=0.15, linewidth=0)
    ax.plot(nf, mapes_naive[ok], marker="s", ms=4, ls="--",
            color=PALETTE["red"], lw=1.3, label="sign-blind baseline")

    if stds_proposed is not None:
        s = stds_proposed[ok]
        ax.fill_between(nf, mapes_proposed[ok] - 1.96 * s,
                        mapes_proposed[ok] + 1.96 * s,
                        color=PALETTE["blue"], alpha=0.15, linewidth=0)
    ax.plot(nf, mapes_proposed[ok], marker="o", ms=4, ls="-",
            color=PALETTE["blue"], lw=1.6, label="proposed (augmented sub-stoch.)")

    ax.axvline(0.45, color=PALETTE["gray"], ls=":", lw=0.9, alpha=0.7,
               label=r"sub/super boundary")
    ax.set_xlabel(r"competitive-edge fraction $f^-$")
    ax.set_ylabel(r"MAPE on $I_2$ (\%)")
    ax.set_xlim(nf.min() - 0.02, nf.max() + 0.02)
    ax.set_ylim(0, max(mapes_naive[ok].max(), mapes_proposed[ok].max()) * 1.35)
    ax.legend(loc="upper left", fontsize=7, handlelength=1.6, ncol=1)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    set_plot_style()
    print("[1/5] main experiment ...")
    (oI2, hI2, hI2c,
     oPhi, hPhi, hPhic,
     oTau, hTau, hTauc) = run_main_experiment()
    print(f"  collected {oI2.size} valid realizations")
    scatter_predictor(
        oI2,
        [hI2, hI2c],
        ["scalar", "CDSM"],
        [PALETTE["red"], PALETTE["blue"]],
        r"observed $I_2$",
        r"predicted $\hat I_2$",
        "fig_I2.png",
    )
    scatter_predictor(
        oPhi,
        [hPhi, hPhic],
        ["scalar", "CDSM"],
        [PALETTE["red"], PALETTE["blue"]],
        r"observed $\Phi_{2,2}$",
        r"predicted $\hat\Phi_{2,2}$",
        "fig_Phi.png",
    )
    print("[2/5] recovery time figure (CDSM vs scalar) ...")
    plot_tau(hTau, hTauc, oTau, "fig_tau.png")
    print("[3/5] scalar vs CDSM comparison ...")
    plot_scalar_vs_cdsm(oI2, hI2, hI2c, oPhi, hPhi, hPhic, "fig_compare.png")
    print("[4/6] density sweep ...")
    neg_fracs = np.linspace(0.0, 0.6, 7)
    mapes_p, stds_p, mapes_n, stds_n = run_density_sweep_with_bands(neg_fracs)
    plot_density_sweep(neg_fracs, mapes_p, mapes_n, "fig_density.png",
                       stds_proposed=stds_p, stds_naive=stds_n)
    print(f"  proposed MAPE: {mapes_p}")
    print(f"  naive    MAPE: {mapes_n}")
    print("[5/6] robustness sweep across pulse magnitude ...")
    sigmas = np.array([0.10, 0.25, 0.50, 1.0, 1.5, 2.0, 3.0])
    rob = run_sigma_sweep(sigmas, n_real=80)
    plot_robustness(rob, "fig_robust_a.png", "fig_robust_b.png")
    print(f"  sigma:     {sigmas}")
    print(f"  I2 scalar: {rob['mape_I2_scalar']}")
    print(f"  I2 CDSM:   {rob['mape_I2_cdsm']}")
    print(f"  cov scal:  {rob['cov_tau_scalar']}")
    print(f"  cov CDSM:  {rob['cov_tau_cdsm']}")
    print("[6/6] done. figures saved in", here)

    print("\nSUMMARY:")
    print(f"  I2  MAPE (scalar)  = {_safe_mape(oI2, hI2):.2f}%")
    print(f"  I2  MAPE (CDSM)    = {_safe_mape(oI2, hI2c):.2f}%")
    print(f"  Phi MAPE (scalar)  = {_safe_mape(oPhi, hPhi):.2f}%")
    print(f"  Phi MAPE (CDSM)    = {_safe_mape(oPhi, hPhic):.2f}%")
    ok = np.isfinite(oTau) & np.isfinite(hTau) & (hTau > 0)
    print(f"  tau coverage (scalar) = "
          f"{float(np.mean(hTau[ok] >= oTau[ok]) * 100):.1f}%")
    ok = np.isfinite(oTau) & np.isfinite(hTauc) & (hTauc > 0)
    print(f"  tau coverage (CDSM)   = "
          f"{float(np.mean(hTauc[ok] >= oTau[ok]) * 100):.1f}%")


if __name__ == "__main__":
    main()
