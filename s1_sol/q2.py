"""Question 2 functions"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
import matplotlib.cm as cm      
import matplotlib.colors as mcolors
import pandas as pd
from scipy.optimize import curve_fit

# Question 2) (i)
def plot_unbinned_mle(df):
    """Performs an unbinned ML fit for each E_0 using iminuit and plots the inidividual fits of each E_0 overlaid."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    #creating x array and an empty y array for the total pdf to fill
    x_arr = np.linspace(-20, 20, 1000)
    total_summed_pdf = np.zeros_like(x_arr)

    fit_results = {}

    #defining function iminuit can read
    def gaussian_pdf(x, mu, sigma):
        return norm.pdf(x, loc=mu, scale=sigma)

    #setting histogram plotting colors
    unique_energies = sorted(df['E_true'].unique())
    norm_color = mcolors.Normalize(vmin=min(unique_energies), vmax=max(unique_energies))
    cmap = cm.viridis

    #defining common bins
    common_bins = np.linspace(-20, 20, 50)
    bin_width = common_bins[1] - common_bins[0]

    #iterating over each E_0 group
    for E_0, group in df.groupby('E_true'):
        data = group['E_rec'].values

        nll = UnbinnedNLL(data, gaussian_pdf)

        # providing minuit with guesses of sample stats
        m = Minuit(nll, mu=np.mean(data), sigma=np.std(data))

        #running minimisation and error analysis
        m.migrad()
        m.hesse()

        #extracting best fit values
        mu_hat = m.values['mu']
        sigma_hat = m.values['sigma']
        mu_error = m.errors['mu']
        sigma_error = m.errors['sigma']

        fit_results[E_0] = {
            'mu': (mu_hat, mu_error), 
            'sigma': (sigma_hat, sigma_error)
        }

        residuals = data - E_0

        #plotting left-hand graph
        color = cmap(norm_color(E_0))
        counts, bins, _ = ax1.hist(residuals, bins=common_bins, histtype='step', label=f'$E_0={E_0}$', alpha=0.8, color=color)
        n_events = len(data)

        pdf_y = norm.pdf(x_arr, loc=(mu_hat - E_0), scale = sigma_hat)

        #scale the pdf to match the histogram area
        scaled_pdf = pdf_y * (n_events * bin_width)

        #plotting the curves on the left-hand graph
        ax1.plot(x_arr, scaled_pdf, linestyle='--', linewidth=1, alpha = 0.8, color=color)

        total_summed_pdf += scaled_pdf

    #plotting right-hand graph
    all_diffs = df['E_rec'] - df['E_true']
    ax2.hist(all_diffs, bins=common_bins, histtype='step', color='k', label='Total Data', linewidth=1.5)
    ax2.plot(x_arr, total_summed_pdf, 'r-', label='Sum of Fits', linewidth=1.5)

    #extra formatting
    ax1.set_xlabel(r"$(E - E_0)$ [GeV]")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Individual Fits")
    ax1.legend()
    
    ax2.set_xlabel(r"$(E - E_0)$ [GeV]")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Total Data vs. Sum of Fits")
    ax2.legend()
    
    fig.tight_layout()

    ### converting to pandas df format ###
    energies = sorted(fit_results.keys())
    
    #extract values into lists
    mus = [fit_results[e]['mu'][0] for e in energies]
    mu_errs = [fit_results[e]['mu'][1] for e in energies]
    sigmas = [fit_results[e]['sigma'][0] for e in energies]
    sigma_errs = [fit_results[e]['sigma'][1] for e in energies]
    
    #convert to pandas series
    s_mu = pd.Series(mus, index=energies)
    s_mu_err = pd.Series(mu_errs, index=energies)
    s_sigma = pd.Series(sigmas, index=energies)
    s_sigma_err = pd.Series(sigma_errs, index=energies)
    
    #fold into a tuple
    values = (s_mu, s_mu_err, s_sigma, s_sigma_err)

    return values, fig
    

