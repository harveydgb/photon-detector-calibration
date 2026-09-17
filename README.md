# Photon Detector Calibration

Estimating the calibration and resolution parameters of a photon detector by four
independent statistical routes — sample moments, individual maximum likelihood, a
simultaneous fit, and a non-parametric bootstrap.

**Statistical Methods · MPhil in Data Intensive Science, University of Cambridge (2025/26)**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![SciPy](https://img.shields.io/badge/SciPy-optimize-8caae6)
![iminuit](https://img.shields.io/badge/method-unbinned%20MLE-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## The problem

A detector is calibrated with single photons at seven known energies
`E₀ = [20, 30, 40, 50, 60, 70, 80] GeV`, with 1000 measurements at each point. The measured
energy is normally distributed, `E ~ N(μ_E, σ_E²)`, with a linear response

```
μ_E = λE₀ + Δ
```

and a resolution that combines stochastic, noise and constant terms:

```
(σ_E/E₀)² = (a/√E₀)² + (b/E₀)² + c²
```

The task is to estimate `{λ, Δ, a, b, c}` from 7,000 measurements — and, more importantly, to
quantify how much the answer depends on the estimator chosen.

## Results

| Parameter | Sample estimates + least squares | Individual unbinned MLE | Simultaneous unbinned MLE |
| --- | --- | --- | --- |
| λ | 1.0120 ± 0.0019 | 1.0120 ± 0.0019 | 1.0120 ± 0.0019 |
| Δ | 1.964 ± 0.087 | 1.964 ± 0.087 | 1.964 ± 0.087 |
| a | 0.387 ± 0.060 | 0.387 ± 0.059 | 0.398 ± 0.056 |
| b | 1.127 ± 0.376 | 1.126 ± 0.376 | 1.066 ± 0.384 |
| c | 0.0343 ± 0.0077 | 0.0342 ± 0.0077 | 0.0329 ± 0.0078 |

The linear response parameters `λ` and `Δ` are recovered identically by every method and are
tightly constrained — the calibration itself is unambiguous.

The resolution parameters are the interesting case. `a`, `b` and `c` shift between the
simultaneous fit and the per-point methods, and `b` carries a fractional uncertainty of
roughly 35% throughout. That degeneracy is structural: the `b/E₀` noise term is only
distinguishable from the stochastic `a/√E₀` term at low energy, and with a lowest calibration
point of 20 GeV the data simply cannot separate them cleanly. Pooling all seven energies into
one likelihood tightens `a` and `c` slightly while leaving `b` essentially unconstrained.

A non-parametric bootstrap (2,500 resamples) then tests whether the analytic uncertainties can
be trusted without assuming the model is correct.

## Repository structure

```
.
├── s1_sol/
│   ├── main.py            # Data loading and plot styling
│   ├── q1.py              # Sample estimates and least-squares fitting
│   ├── q2.py              # Individual unbinned MLE fits
│   ├── q3.py              # Simultaneous unbinned MLE fit
│   ├── q4.py              # Non-parametric bootstrap
│   └── mphil.mplstyle
├── notebooks/
│   ├── solution.ipynb     # Full analysis and written discussion
│   └── instructions.ipynb # Problem statement
├── tests/                 # CI and notebook-execution checks
├── figs/                  # Generated figures (PDF)
├── results.json           # Parameter values and uncertainties, all methods
├── sample.csv             # Calibration data — 7,000 measurements
└── pyproject.toml
```

## Analysis sections

| Question | Content |
| --- | --- |
| 1 | Residual distributions, sample estimates per energy, least-squares fit of the resolution model |
| 2 | Independent unbinned maximum-likelihood fit at each energy point |
| 3 | Simultaneous unbinned MLE across all seven energies |
| 4 | Non-parametric bootstrap (2,500 resamples) for distribution-free uncertainties |

## Setup

```bash
pip install -e .
jupyter notebook notebooks/solution.ipynb
```

Run all cells from the repository root. Figures are written to `figs/` as
`Figure1.1.pdf` … `Figure4.2.pdf`, and parameter estimates to `results.json`.

Notebook paths are relative to `notebooks/` (`../sample.csv`,
`../s1_sol/mphil.mplstyle`), so keep the working directory at the repository root.

The Q4 bootstrap takes roughly 1–3 minutes depending on CPU.

## Tests

```bash
pytest tests -v
```

Checks that the notebook executes end to end, that `results.json` is produced in the expected
format, and that every required figure is generated.

## Author

Harvey Bermingham — MPhil in Data Intensive Science, University of Cambridge

## License

Released under the MIT License. See [LICENSE](LICENSE).
