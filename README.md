# S1 Coursework: Photon Detector Calibration Analysis

**MPhil DIS: Statistical Methods for Data Intensive Science (2025/26)**

This project implements a statistical analysis of photon detector calibration data using multiple estimation methods to determine detector performance parameters.

## Usage

1. **Install deps (Python 3.10+):** `pip install -e .`
2. **From the repo root, find** `notebooks/solution.ipynb` **and Run All.**
3. **Outputs land in the repo root:**
   - **Figures:** `figs/Figure1.1.pdf` ... `Figure4.2.pdf`
   - **Results:** `results.json` (parameter values/errors for all methods)
   - **Notebook is executed in-place, keeping outputs in** `notebooks/solution.ipynb`**.**

**Notes:**
- The non-parametric bootstrap (Q4, n_boot=2500) takes ~1-3 minutes depending on CPU.
- Paths in the notebook are relative to `notebooks/` (e.g. `../sample.csv`, `../s1_sol/mphil.mplstyle`), keep the working directory as the repo root when executing.

## Overview

The analysis aims to estimate calibration parameters for a photon detector by fitting models to calibration measurements. The detector was tested at seven energy levels (20, 30, 40, 50, 60, 70, 80 GeV) with 1000 measurements at each level. The measured energy $E$ follows a normal distribution with mean $\mu_E = \lambda E_0 + \Delta$ and width $\sigma_E$ given by a resolution model. The goal is to estimate the parameters $\{\lambda, \Delta, a, b, c\}$ using various statistical methods.

## Project Structure

```
hb747/
├── s1_sol/                    # Main analysis package
│   ├── __init__.py
│   ├── main.py                # Data loading and style configuration
│   ├── q1.py                  # Sample estimates and least squares fitting
│   ├── q2.py                  # Individual unbinned MLE fits
│   ├── q3.py                  # Simultaneous unbinned MLE fit
│   ├── q4.py                  # Non-parametric bootstrap analysis
│   └── mphil.mplstyle         # Matplotlib style file
├── notebooks/
│   ├── instructions.ipynb    # Assignment instructions
│   └── solution.ipynb         # Complete solution and analysis
├── figs/                       # Generated figures (PDF format)
│   ├── Figure1.1.pdf          # Total residual distribution
│   ├── Figure1.2.pdf          # Overlaid residual distributions
│   ├── Figure1.3.pdf          # Sample estimates
│   ├── Figure1.4.pdf          # Sample estimate fits
│   ├── Figure2.1.pdf          # Individual MLE fits
│   ├── Figure2.2.pdf          # Individual fit results
│   ├── Figure3.1.pdf          # Simultaneous MLE fit
│   ├── Figure3.2.pdf          # Parameter comparison
│   ├── Figure4.1.pdf          # Bootstrap histograms
│   └── Figure4.2.pdf          # Bootstrap vs analytical errors
├── sample.csv                  # Input calibration data
├── results.json                # Parameter estimates and uncertainties
├── pyproject.toml             # Package configuration
└── README.md                   # This file
```

## Analysis Methods

**Q1 - Sample Estimates:**
- Calculates sample mean and standard deviation for each energy level
- Fits linear model for mean: $\mu_E = \lambda E_0 + \Delta$
- Fits resolution model for width using least squares
- Uses parametric bootstrap for uncertainty bands (16th/84th percentiles)

**Q2 - Individual MLE Fits:**
- Performs unbinned maximum likelihood estimation for each energy level
- Fits Gaussian distributions independently
- Validates normal distribution assumption
- Results used as input for least squares fitting

**Q3 - Simultaneous MLE Fit:**
- Global unbinned maximum likelihood fit across all data
- Simultaneously estimates all parameters $\{\lambda, \Delta, a, b, c\}$
- Uses Minuit optimization with parameter constraints
- Provides unified parameter estimates

**Q4 - Bootstrap Analysis:**
- Non-parametric bootstrap with 2500 iterations
- Resamples entire dataset with replacement
- Applies all three methods to each bootstrap sample
- Compares bootstrap uncertainties with analytical (Hessian-based) errors
- Reveals non-Gaussian behavior in parameter distributions

## Dependencies

- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `scipy` - Scientific computing (optimization, curve fitting)
- `pandas` - Data manipulation
- `iminuit` - Maximum likelihood estimation

All dependencies are automatically installed when installing the package.

## Results

**Output Files:**
- **Figures** (`figs/` directory): All analysis visualizations saved as PDF files
- **Parameter Estimates** (`results.json`): Contains parameter values and uncertainties for all three methods (`sample_ests`, `individual_fits`, `simultaneous_fit`)

**Key Findings:**
- All three methods produce consistent parameter estimates
- Sample estimate and individual MLE methods show excellent agreement
- Detector shows systematic bias ($\lambda \approx 1.012$, $\Delta \approx 1.96$ GeV)
- Resolution improves at higher energies (fractional error drops from ~11% to ~6%)
- Bootstrap analysis reveals non-Gaussian uncertainties for width parameters

## Technical Details

- **Uncertainty bands**: Error bands use 16th/84th percentiles (equivalent to ±1σ) to handle non-Gaussian distributions
- **Bootstrap iterations**: 2500 bootstrap samples used for uncertainty analysis
- **Optimization**: Minuit package used for MLE with likelihood error definition
- **Parameter constraints**: Parameters $a$, $b$, and $c$ constrained to be positive

## Author

**Harvey Bermingham**  
Email: harveybermingham1@gmail.com  
Cambridge University ID: hb747@cam.ac.uk

## License

See LICENSE file for details.

## Acknowledgments

Coursework for MPhil DIS: Statistical Methods for Data Intensive Science, 2025/26.

## Use of Generative Tools

This project has utlised auto-generative tools in the development of the analysis for this project.

Example prompts used for this project:
- Generate code for zoomed in snapshots of these parameters within the figure
- Create a general README.md template structure for this project
- Generate doc-strings for this function
- Review overall project structure for completeness, consistency and best practice

