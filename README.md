# Stability and Resilience of Competitive Networked Systems

Reference Python implementation accompanying:

> **Predicting Temporal Stability and Resilience of Competitive Networked Systems via Sub/Super-Stochastic Matrices.** S. Yan. Submitted to *IEEE Transactions on Cybernetics*, 2026.

The code reproduces every figure of Section VI of the paper, including the
real-data cross-check on the Cedar Creek BioDIV (e120) grassland experiment
used by Isbell et al. (Nature, 2026).

## Contents

| File | Role |
|---|---|
| `simulation.py` | Synthetic-network experiments: contraction certificate, scalar and CDSM predictors of `I_2`, `Φ_{2,x}`, `τ_ε`, robustness sweep across pulse magnitude, antagonism-density sweep |
| `biodiv_validation.py` | Field-data cross-check on the BioDIV (e120) dataset: per-block extraction of `(Ω_2, Δ_2)`, empirical CDSM envelope, scatter plots of observed vs. predicted `I_2` and `Φ_{2,2}` |
| `download_data.py` | One-shot fetch of the BioDIV CSV from the Environmental Data Initiative |
| `requirements.txt` | Python dependencies |

## Quick start

```bash
git clone https://github.com/<your-handle>/StabilityResilience_SubSuperStochastic.git
cd StabilityResilience_SubSuperStochastic
pip install -r requirements.txt

# synthetic experiments (Figs. 1-6 of the paper)
python simulation.py

# real-data cross-check (Fig. 7)
python download_data.py
python biodiv_validation.py
```

`simulation.py` writes the figures `fig_I2.png`, `fig_Phi.png`, `fig_tau.png`,
`fig_compare.png`, `fig_robust_a.png`, `fig_robust_b.png`, and `fig_density.png`
to the current directory. `biodiv_validation.py` writes `fig_biodiv_I2_*.png`
and `fig_biodiv_Phi_*.png`.

## Algorithmic core

The package implements the *Cycle-Decaying Stochastic Matrix* (CDSM)
introduced in Section III of the paper, alongside the closed-form scalar
predictors. Both forms are exposed:

```python
import simulation as sim

I2_hat       = sim.predict_I2(Omega2, Delta2, P, T)            # scalar
I2_hat_cdsm  = sim.predict_I2_cdsm(Omega2, eta, P, T)          # profile-aware
Phi_hat      = sim.predict_Phi(Omega2, Delta2, x)
Phi_hat_cdsm = sim.predict_Phi_cdsm(Omega2, eta, x)
tau_hat      = sim.predict_tau(Omega2, Delta2, eps)
tau_hat_cdsm = sim.predict_tau_cdsm(Omega2, eta, eps)
```

where `eta` is the per-step decay profile of the empirical CDSM envelope
(returned by `sim.cdsm_envelope_empirical`).

## Data

The BioDIV (e120) aboveground-biomass CSV is **not** redistributed with this
repository. `download_data.py` fetches it from the authoritative EDI source
on first run:

> Tilman, D. "Plant Aboveground Biomass Data" (knb-lter-cdr.273.11).
> <https://doi.org/10.6073/pasta/27ddb5d8aebe24db99caa3933e9bc8e2>.
> Licensed CC BY 4.0.

Cite the dataset whenever the downloaded CSV is used.

## Citing this code

If you use this code in academic work, please cite the accompanying paper.
A BibTeX entry will be added after acceptance.

## License

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0 (held by Cedar Creek LTER /
EDI; see notice in `LICENSE`).
