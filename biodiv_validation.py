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
    numsp_map = plot_year.attrs.get("NumSp", {})
    if not isinstance(numsp_map, pd.Series):
        numsp_map = pd.Series(numsp_map)
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
        rows.append(dict(plot=plot, NumSp=numsp_map.get(plot, np.nan),
                         Yn=Yn, Omega2=Omega2, Delta2=Delta2,
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


def _panel(ax, obs, h_scalar, h_cdsm, numsp, ylabel, show_legend=False):
    """Draw a single observed-vs-predicted panel with scalar + CDSM overlay,
    colour-coded by NumSp."""
    import matplotlib.cm as cm
    levels = sorted(set(int(x) for x in numsp if np.isfinite(x)))
    cmap = cm.get_cmap("viridis", len(levels))
    level_color = {lv: cmap(i) for i, lv in enumerate(levels)}

    pooled_obs = obs[np.isfinite(obs)]
    pooled_pred = np.concatenate([h_scalar[np.isfinite(h_scalar)],
                                   h_cdsm[np.isfinite(h_cdsm)]])
    lo = float(min(pooled_obs.min(), pooled_pred.min()))
    hi = float(max(pooled_obs.max(), pooled_pred.max()))
    pad = (hi - lo) * 0.07
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--",
            lw=0.8, color=sim.PALETTE["gray"], zorder=1)

    for ns in levels:
        sel = (numsp == ns)
        col = level_color[ns]
        ok_s = sel & np.isfinite(h_scalar) & np.isfinite(obs)
        ok_c = sel & np.isfinite(h_cdsm) & np.isfinite(obs)
        if ok_s.any():
            ax.scatter(obs[ok_s], h_scalar[ok_s], s=24, facecolors="none",
                       edgecolors=col, linewidths=0.9, zorder=2)
        if ok_c.any():
            ax.scatter(obs[ok_c], h_cdsm[ok_c], s=22, c=[col], linewidth=0,
                       alpha=0.85, zorder=3)

    m_s = mape(obs, h_scalar)
    m_c = mape(obs, h_cdsm)
    txt = (r"\textsf{scalar: " + f"{m_s:.0f}" + r"\%}"
           + "\n" + r"\textsf{CDSM: " + f"{m_c:.0f}" + r"\%}")
    sim._inset_metric_box(ax, txt, loc="upper left")

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"observed " + ylabel)
    ax.set_ylabel(r"predicted " + ylabel)
    ax.grid(True)

    if show_legend:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor=sim.PALETTE["edge"], markeredgewidth=0.9,
                   markersize=5, label="scalar"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=sim.PALETTE["edge"], markersize=5,
                   label="CDSM"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=7,
                  handletextpad=0.3, borderaxespad=0.4)
    return levels, level_color


def make_panel_figure(metrics: pd.DataFrame, which: str, fname: str) -> None:
    """Generate one self-contained subfigure (I_2 or Phi_{2,2}) with an attached
    discrete NumSp colour bar on the right edge."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    obs_col = "I2_obs"   if which == "I2" else "Phi2_obs"
    sc_col  = "I2_scalar" if which == "I2" else "Phi2_scalar"
    cd_col  = "I2_cdsm"   if which == "I2" else "Phi2_cdsm"
    ylab    = r"$I_2$"    if which == "I2" else r"$\Phi_{2,2}$"

    obs = metrics[obs_col].to_numpy(dtype=float)
    sc  = metrics[sc_col].to_numpy(dtype=float)
    cd  = metrics[cd_col].to_numpy(dtype=float)
    numsp = metrics["NumSp"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(2.7, 2.4))
    levels, _ = _panel(ax, obs, sc, cd, numsp, ylab, show_legend=False)

    # marker convention (scalar vs CDSM) inside the plot
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=sim.PALETTE["edge"], markeredgewidth=0.9,
               markersize=4, label="scalar"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=sim.PALETTE["edge"], markersize=4,
               label="CDSM"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6,
              handletextpad=0.2, borderaxespad=0.4)

    # discrete NumSp colour bar attached to the right edge
    cmap = cm.get_cmap("viridis", len(levels))
    norm = mcolors.BoundaryNorm(
        boundaries=[i - 0.5 for i in range(len(levels) + 1)],
        ncolors=len(levels))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.055, pad=0.04, ticks=range(len(levels)))
    cbar.ax.set_yticklabels([str(int(lv)) for lv in levels])
    cbar.set_label(r"NumSp", rotation=90, labelpad=4)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=7, width=0.6, length=2)

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
    blocks = bootstrap_aggregated_timeseries(py, block_size=10,
                                             n_per_richness=20, seed=0)
    blocks.attrs["NumSp"] = pd.Series(blocks.attrs["NumSp_block"])
    print(f"  {blocks.shape[0]} aggregated trajectories x {blocks.shape[1]} years")
    print("[3/4] extracting per-block metrics + applying predictors ...")
    metrics = per_plot_metrics(blocks)
    metrics = apply_predictors(metrics)
    print(f"  {len(metrics)} blocks usable for prediction")
    print("[4/4] generating subfigures ...")
    make_panel_figure(metrics, "I2",  "fig_biodiv_a.png")
    make_panel_figure(metrics, "Phi", "fig_biodiv_b.png")

    print("\nSUMMARY (BioDIV real-data validation, seed=0 displayed):")
    print(f"  blocks: {len(metrics)} (bootstrap resamples; each plot reused ~6x)")
    print(f"  I2  scalar MAPE = {mape(metrics['I2_obs'], metrics['I2_scalar']):.2f}%")
    print(f"  I2  CDSM   MAPE = {mape(metrics['I2_obs'], metrics['I2_cdsm']):.2f}%")
    print(f"  Phi scalar MAPE = {mape(metrics['Phi2_obs'], metrics['Phi2_scalar']):.2f}%")
    print(f"  Phi CDSM   MAPE = {mape(metrics['Phi2_obs'], metrics['Phi2_cdsm']):.2f}%")

    # multi-seed robustness check (10 seeds)
    print("\nROBUSTNESS (10 random seeds, mean +/- std):")
    rows = []
    for seed in range(10):
        bk = bootstrap_aggregated_timeseries(py, block_size=10,
                                             n_per_richness=20, seed=seed)
        bk.attrs["NumSp"] = pd.Series(bk.attrs["NumSp_block"])
        mm = apply_predictors(per_plot_metrics(bk))
        rows.append((
            mape(mm["I2_obs"], mm["I2_scalar"]),
            mape(mm["I2_obs"], mm["I2_cdsm"]),
            mape(mm["Phi2_obs"], mm["Phi2_scalar"]),
            mape(mm["Phi2_obs"], mm["Phi2_cdsm"]),
        ))
    arr = np.array(rows)
    for i, lbl in enumerate(["I2  scalar", "I2  CDSM  ", "Phi scalar", "Phi CDSM  "]):
        print(f"  {lbl} MAPE = {arr[:, i].mean():.1f}% +/- {arr[:, i].std():.1f}%"
              f"  (range {arr[:, i].min():.1f}-{arr[:, i].max():.1f})")
    print(f"  Isbell 2026 baseline (ecosystem-level): I2 = 1.1%, Phi = 3.0%")


if __name__ == "__main__":
    main()
