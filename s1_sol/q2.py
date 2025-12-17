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

# Question 2) (i)

def unbinned_mle(df):
    """
    Uses the unbinned MLE to fit Gaussian distributions to the measured energy values
    for each E_0 group. Fits are performed on the actual E values (not residuals).
    
    Inputs
    df: pandas dataframe of all detector measurements with columns 'E_rec' and 'E_true'

    Returns
    mle_fitted_values: tuple in form of (mu_mle, mu_mle_err, sigma_mle, sigma_mle_err)
                       where each element is a pandas Series indexed by E_0 values
    """
    fit_results = {}

    #defining function iminuit can read
    def gaussian_pdf(x, mu, sigma):
        return norm.pdf(x, loc=mu, scale=sigma)

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

    ### converting to pandas df format ###
    energies = sorted(fit_results.keys())
    
    #extract values into lists
    mus = [fit_results[e]['mu'][0] for e in energies]
    mu_errs = [fit_results[e]['mu'][1] for e in energies]
    sigmas = [fit_results[e]['sigma'][0] for e in energies]
    sigma_errs = [fit_results[e]['sigma'][1] for e in energies]
    
    #convert to pandas series
    mu_mle = pd.Series(mus, index=energies)
    mu_mle_err = pd.Series(mu_errs, index=energies)
    sigma_mle = pd.Series(sigmas, index=energies)
    sigma_mle_err = pd.Series(sigma_errs, index=energies)
    
    #fold into a tuple
    mle_fitted_values = (mu_mle, mu_mle_err, sigma_mle, sigma_mle_err)

    return mle_fitted_values

def unbinned_mle_and_histogram_plot(df, mle_fitted_values):
    """Plots histograms of the distributions of residuals from each E_0 value
    alongside the MLE fitted and scaled distributions. Also plots a second graph
    which shows the total histogram and sum of all fitted PDFs.
    
    Inputs
    df: pandas dataframe of all detector measurements
    mle_fitted_values: tuple in form of (mu_mle, mu_mle_err, sigma_mle, sigma_mle_err)
    
    Returns
    fig: matplotlib figure object"""

    #unpacking
    mu_mle, mu_mle_err, sigma_mle, sigma_mle_err = mle_fitted_values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    plt.suptitle("Figure 2.1: Individual MLE Sample Parameter Fitting", y=0.95)

    #creating x array and an empty y array for the total pdf to fill
    x_arr = np.linspace(-20, 20, 1000)
    total_summed_pdf = np.zeros_like(x_arr)
    
    #setting histogram plotting colors
    unique_energies = sorted(df['E_true'].unique())
    norm_color = mcolors.Normalize(vmin=min(unique_energies), vmax=max(unique_energies))
    cmap = cm.viridis

    #defining common bins
    diff = df['E_rec'] - df['E_true']
    common_bins = np.linspace(np.floor(diff.min()), np.ceil(diff.max()), 50)
    bin_width = common_bins[1] - common_bins[0]

    #iterating over each E_0 group
    for E_0, group in df.groupby('E_true'):
        
        #calculating residual from respective E_0 values
        data = group['E_rec'].values
        residuals = data - E_0

        #plotting left-hand graph
        color = cmap(norm_color(E_0))
        counts, bins, _ = ax1.hist(residuals, bins=common_bins, histtype='step', label=f'$E_0={E_0}$', alpha=0.8, color=color)
        n_events = len(data)


        pdf_y = norm.pdf(x_arr, loc=(mu_mle[E_0] - E_0), scale = sigma_mle[E_0])

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

    return fig

def unbinned_mle_fit_and_plot(df):
    """Uses the unbinned MLE to fit Gaussian distributions to the measured energy values
    for each E_0 group. Then plots histograms of the distributions of residuals from each E_0 value
    alongside the MLE fitted and scaled distributions. Also plots a second graph
    which shows the total histogram and sum of all fitted PDFs.
    
    Inputs
    df: pandas dataframe of all detector measurements
    
    Returns
    fig: matplotlib figure object
    mle_fitted_values: tuple in form of (mu_mle, mu_mle_err, sigma_mle, sigma_mle_err)
                       where each element is a pandas Series indexed by E_0 values"""

    mle_fitted_values = unbinned_mle(df)
    fig = unbinned_mle_and_histogram_plot(df, mle_fitted_values)

    return fig, mle_fitted_values


