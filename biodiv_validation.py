"""
Validation on the Cedar Creek BioDIV (e120) experiment.

Uses the same 27-year grassland aboveground-biomass dataset as Isbell et al.
(Nature, 2026). For each plot:
  * the long-term reference Y_n is the mean over non-perturbation years
  * the 2002 extreme wet event is the perturbation cycle used to extract
    (Omega_2, Delta_2)
  * the empirical CDSM envelope is built from the 2002 -> 2007 trajectory
  * scalar and CDSM predictors are applied to predict
      - long-term temporal stability I_2 (1996-2021 full record)
      - 2-step resilience Phi_{2,2} measured at 2004

Compares head-to-head with the Isbell baseline (MAPE ~ 1.1% / 3.0%).

Outputs:
  fig_biodiv_I2.png   - scatter observed vs predicted I_2
  fig_biodiv_Phi.png  - scatter observed vs predicted Phi_{2,2}
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Re-use predictors / style from the main simulation module
import simulation as sim


DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "ref", "biodiv_e120_biomass.csv",
)

# Isbell 2026 perturbation/resilience anchors. The BioDIV plots take roughly
# four years to establish their mature standing biomass (~1996-1999 transient
# growth phase). We therefore restrict the long-term stability window to the
# post-establishment regime, in line with the diversity-interactions models
# fitted in Isbell et al. that effectively absorb the establishment transient.
PERTURB_YEAR = 2002       # extreme wet event
RECOVERY_YEAR = 2003      # year after
RESILIENCE_YEAR = 2004    # 2 years after (used for Phi_{2,2})
TOBS_START, TOBS_END = 2000, 2021    # post-establishment 22-year window
P_OBS = 2                 # 2002 wet + 2021 drought, per Isbell
P_INTERVAL = (TOBS_END - TOBS_START + 1) // P_OBS  # ~11 years between events


def load_plot_timeseries(path: str = DATA_PATH) -> pd.DataFrame:
    """Return a (plot x year) matrix of yearly aboveground biomass."""
    df = pd.read_csv(path, low_memory=False)
    plot_year = df.groupby(["Plot", "Year"])["Biomass (g/m2)"].sum().unstack()
    plot_year = plot_year.loc[
        :, (plot_year.columns >= TOBS_START) & (plot_year.columns <= TOBS_END)
    ]
    # keep only plots with full coverage of the perturbation and recovery years
    must_have = [PERTURB_YEAR - 1, PERTURB_YEAR, RECOVERY_YEAR, RESILIENCE_YEAR]
    plot_year = plot_year.dropna(subset=must_have)
    # attach richness for later treatment-level aggregation
    plot_numsp = df.drop_duplicates(["Plot"]).set_index("Plot")["NumSp"]
    plot_year.attrs["NumSp"] = plot_numsp
    return plot_year


def bootstrap_aggregated_timeseries(plot_year: pd.DataFrame,
                                    block_size: int = 5,
                                    n_per_richness: int = 6,
                                    seed: int = 0) -> pd.DataFrame:
    """Group plots by NumSp and create averaged trajectories of `block_size`
    randomly drawn plots each, repeated `n_per_richness` times per NumSp level.

    The averaging reduces the per-plot demographic noise that Isbell et al.
    absorb via diversity-interactions models, while preserving the climate
    signal (2002 wet, 2021 drought) that is common across plots.
    """
    rng = np.random.default_rng(seed)
    plot_numsp = plot_year.attrs["NumSp"]
    rows = []
    for ns in sorted(plot_numsp.unique()):
        if ns == 0:
            continue
        members = plot_year.index[plot_year.index.map(plot_numsp).fillna(-1) == ns]
        if len(members) < block_size:
            continue
        for k in range(n_per_richness):
            sel = rng.choice(members, size=block_size, replace=False)
            traj = plot_year.loc[sel].mean(axis=0)
            rows.append((f"R{ns}_b{k}", ns, traj))
    out = pd.DataFrame({lbl: tr for lbl, ns, tr in rows}).T
    out.index.name = "block"
    out.attrs["NumSp_block"] = {lbl: ns for lbl, ns, _ in rows}
    return out


def per_plot_metrics(plot_year: pd.DataFrame):
    """Extract observed (Omega_2, Delta_2, I_2, Phi_{2,2}) and the CDSM envelope
    for every plot in plot_year."""
    rows = []
    for plot, ts in plot_year.iterrows():
        ts = ts.dropna()
        # Y_n: mean over non-perturbation years (exclude 2002 +/- 1 and 2021 +/- 1)
        excluded = {2001, 2002, 2003, 2020, 2021}
        Yn = ts[~ts.index.isin(excluded)].mean()
        if not np.isfinite(Yn) or Yn <= 0:
            continue
        Ye = ts[PERTURB_YEAR]
        Y1 = ts[RECOVERY_YEAR]
        Y2 = ts[RESILIENCE_YEAR]
        # complement measures (Isbell D5-D8)
        Omega2 = 1 - abs(Yn - Ye) / Yn
        denom = abs(Yn - Ye)
        Delta2 = 1 - abs(Yn - Y1) / denom if denom > 1e-9 else np.nan
        I2_obs = 1 - ts.std() / ts.mean() if ts.mean() > 0 else np.nan
        Phi2_obs = 1 - abs(Yn - Y2) / Yn
        # empirical CDSM envelope from 2002 -> min(2007, last available)
        cycle_years = [y for y in range(PERTURB_YEAR, min(2008, ts.index.max()) + 1)
                       if y in ts.index]
        if len(cycle_years) < 3 or denom <= 1e-9:
            eta = None
        else:
            cumdev = np.array([abs(Yn - ts[y]) / denom for y in cycle_years])
            cumdev[0] = 1.0  # by definition
            menv = np.maximum.accumulate(cumdev[::-1])[::-1]
            eta = np.clip(menv[1:] / np.maximum(menv[:-1], 1e-12), 1e-6, 1.0)
            # pad to length P_INTERVAL for predictor consistency
            if len(eta) < P_INTERVAL:
                eta = np.concatenate([eta, np.full(P_INTERVAL - len(eta), eta[-1])])
            else:
                eta = eta[:P_INTERVAL]
        rows.append(dict(plot=plot, Yn=Yn, Omega2=Omega2, Delta2=Delta2,
                         I2_obs=I2_obs, Phi2_obs=Phi2_obs, eta=eta))
    return pd.DataFrame(rows)


def apply_predictors(df: pd.DataFrame):
    """Append scalar and CDSM predicted I_2 and Phi_{2,2} columns."""
    T_obs = TOBS_END - TOBS_START + 1
    df = df.copy()
    df["I2_scalar"] = df.apply(
        lambda r: sim.predict_I2(r["Omega2"], r["Delta2"], P_INTERVAL, T_obs)
        if np.isfinite(r["Delta2"]) and 0 < r["Delta2"] < 1 else np.nan, axis=1)
    df["I2_cdsm"] = df.apply(
        lambda r: sim.predict_I2_cdsm(r["Omega2"], r["eta"], P_INTERVAL, T_obs)
        if r["eta"] is not None and 0 < r["Omega2"] < 1 else np.nan, axis=1)
    df["Phi2_scalar"] = df.apply(
        lambda r: sim.predict_Phi(r["Omega2"], r["Delta2"], 2)
        if np.isfinite(r["Delta2"]) and 0 < r["Delta2"] < 1 else np.nan, axis=1)
    df["Phi2_cdsm"] = df.apply(
        lambda r: sim.predict_Phi_cdsm(r["Omega2"], r["eta"], 2)
        if r["eta"] is not None and 0 < r["Omega2"] < 1 else np.nan, axis=1)
    return df


def mape(o, h):
    o = np.asarray(o, dtype=float)
    h = np.asarray(h, dtype=float)
    ok = np.isfinite(o) & np.isfinite(h) & (np.abs(o) > 1e-6)
    return float(np.mean(np.abs((h[ok] - o[ok]) / o[ok])) * 100) if ok.any() else float("nan")


def scatter_with_diag(o, h, xlabel, ylabel, fname, color):
    ok = np.isfinite(o) & np.isfinite(h)
    o, h = o[ok], h[ok]
    if o.size == 0:
        return
    m = mape(o, h)
    r2 = float(np.corrcoef(o, h)[0, 1] ** 2)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.scatter(o, h, s=14, alpha=0.55, color=color, linewidth=0,
               rasterized=True)
    lo, hi = float(min(o.min(), h.min())), float(max(o.max(), h.max()))
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--",
            lw=0.9, color=sim.PALETTE["gray"], label="$y=x$")
    # regression line + 95% band
    sim._regression_band(ax, o, h, color=color)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    sim._inset_metric_box(
        ax,
        f"MAPE = {m:.2f}" + r"\%" + f",  $R^2$ = {r2:.2f}\nn = {o.size} plots",
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def main():
    sim.set_plot_style()
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    print(f"[1/4] loading BioDIV plot-year biomass from {DATA_PATH} ...")
    py = load_plot_timeseries()
    print(f"  retained {py.shape[0]} plots x {py.shape[1]} years "
          f"({py.columns.min()}-{py.columns.max()})")
    print("[2/4] aggregating into subtreatment-block trajectories ...")
    blocks = bootstrap_aggregated_timeseries(py, block_size=5,
                                             n_per_richness=6, seed=0)
    blocks.attrs["NumSp"] = pd.Series(blocks.attrs["NumSp_block"])
    print(f"  {blocks.shape[0]} aggregated trajectories x {blocks.shape[1]} years")
    print("[3/4] extracting per-block metrics + applying predictors ...")
    metrics = per_plot_metrics(blocks)
    metrics = apply_predictors(metrics)
    print(f"  {len(metrics)} blocks usable for prediction")
    print("[4/4] generating figures ...")
    scatter_with_diag(metrics["I2_obs"].values, metrics["I2_scalar"].values,
                      r"observed $I_2$", r"scalar $\hat I_2$",
                      "fig_biodiv_I2_scalar.png", sim.PALETTE["red"])
    scatter_with_diag(metrics["I2_obs"].values, metrics["I2_cdsm"].values,
                      r"observed $I_2$", r"CDSM $\hat I_2$",
                      "fig_biodiv_I2_cdsm.png", sim.PALETTE["blue"])
    scatter_with_diag(metrics["Phi2_obs"].values, metrics["Phi2_scalar"].values,
                      r"observed $\Phi_{2,2}$", r"scalar $\hat\Phi_{2,2}$",
                      "fig_biodiv_Phi_scalar.png", sim.PALETTE["red"])
    scatter_with_diag(metrics["Phi2_obs"].values, metrics["Phi2_cdsm"].values,
                      r"observed $\Phi_{2,2}$", r"CDSM $\hat\Phi_{2,2}$",
                      "fig_biodiv_Phi_cdsm.png", sim.PALETTE["blue"])

    print("\nSUMMARY (BioDIV real-data validation):")
    print(f"  plots: {len(metrics)}")
    print(f"  I2  scalar MAPE = {mape(metrics['I2_obs'], metrics['I2_scalar']):.2f}%")
    print(f"  I2  CDSM   MAPE = {mape(metrics['I2_obs'], metrics['I2_cdsm']):.2f}%")
    print(f"  Phi scalar MAPE = {mape(metrics['Phi2_obs'], metrics['Phi2_scalar']):.2f}%")
    print(f"  Phi CDSM   MAPE = {mape(metrics['Phi2_obs'], metrics['Phi2_cdsm']):.2f}%")
    print(f"  Isbell 2026 baseline (ecosystem-level): I2 = 1.1%, Phi = 3.0%")


if __name__ == "__main__":
    main()
