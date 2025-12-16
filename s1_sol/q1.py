"""Question 1 functions"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.cm as cm      
import matplotlib.colors as mcolors

# Question 1) (i)
def plot_total_hist(df):
    """
    Plotting Histogram of all E - E_0 values.
    """
    
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    
    # Calculate E - E_0
    diff = df['E_rec'] - df['E_true']
    
    # Plot histogram
    ax.hist(diff, bins=50, histtype='step',color='black')
    ax.set_xlabel(r"$(E - E_0)$ [GeV]")
    ax.set_ylabel('Frequency')
    ax.set_title(f"Figure 1.1: Distribution of Residuals", pad=10)

    return fig, ax

# Question 1) (ii)
def plot_overlapping_hist(df):
    """
    Plotting overlapping histograms of each set of (E - E_0) across each of 
    the E_0 values

    Inputs
    df: pandas dataframe of all detector measurements.

    Returns
    fig: figure of overlappping distributions
    """
    
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    
    #setting histogram plotting colors
    unique_energies = sorted(df['E_true'].unique())
    norm_color = mcolors.Normalize(vmin=min(unique_energies), vmax=max(unique_energies))
    cmap = cm.viridis

    #defining common bins
    diff = df['E_rec'] - df['E_true']
    common_bins = np.linspace(np.floor(diff.min()), np.ceil(diff.max()), 50)

    # plotting hist differences across each E_0
    for E_0, group in df.groupby('E_true'):
        diff = group['E_rec'] - E_0
        color = cmap(norm_color(E_0))
        ax.hist(diff, bins=common_bins, histtype='step',color=color, label=f'$E_0={E_0}$')
    
    ax.set_xlabel(r"$(E - E_0)$ [GeV]")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Figure 1.2: Overlayed Residual Distributions", pad=10)
    ax.legend()
    
    return fig

# Question 1) (iii)
def calculate_sample_estimates(df):
    """
    Calculating the samples estimates and associated estimated 
    errors of the mean and standard deviation.
    Inputs
    df: pandas dataframe of all detector measurements.

    Returns
    sample_estimate_values: tuple in form of (mu_samp, mu_error, sigma_samp, sigma_error)
    """
    
    grouped = df.groupby('E_true')['E_rec']

    #calculating samples values and associated errors
    mu_samp = grouped.mean()
    sigma_samp = grouped.std(ddof=1)
    N = grouped.count()

    mu_error = sigma_samp/ (N ** 0.5)
    sigma_error = sigma_samp/( (2*(N-1)) ** 0.5)

    sample_estimate_values = (mu_samp, mu_error, sigma_samp, sigma_error)
    return sample_estimate_values

def plot_sample_estimates(sample_estimate_values):
    """Takes samples estimate values and plots on two separate histograms.
    Returns fig.
    
    Inputs
    sample_estimate_values: tuple in form of (mu_samp, mu_error, sigma_samp, sigma_error)
    
    Returns
    fig: plots of mean and standard deviation of energy measurements."""

    #unpacking
    mu_samp, mu_error, sigma_samp, sigma_error = sample_estimate_values

    #plotting subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    plt.suptitle("Figure 1.3: Sample Estimates", y=0.95)
    
    #samples mean
    ax1.errorbar(mu_samp.index, mu_samp, yerr=mu_error, fmt='o', capsize=4,color='black')
    ax1.set_xlabel(r"$E_0$ [GeV]")
    ax1.set_ylabel(r"Mean Energy $\hat{\mu}_{\rm samp}$ [GeV]")
    ax1.set_title(r"Sample Mean vs $E_0$")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    #sample standard deviation
    ax2.errorbar(sigma_samp.index, sigma_samp, yerr=sigma_error, fmt='o', capsize=4,color='black')
    ax2.set_xlabel(r"$E_0$ [GeV]")
    ax2.set_ylabel(r"Standard Deviation $\hat{\sigma}_{\rm samp}$ [GeV]")
    ax2.set_title(r"Sample Standard Deviation vs $E_0$")
    ax2.grid(True, linestyle='--', alpha=0.5)

    return fig

def calculate_and_plot_sample_estimates(df):
    
    sample_estimate_values = calculate_sample_estimates(df)
    fig = plot_sample_estimates(sample_estimate_values)

    return fig, sample_estimate_values

# Question 1) (iv)

def mean_func(E_0, lb, dE):
    return lb * E_0 + dE

def sigma_func(E_0, a, b, c):
    return np.sqrt(np.abs((E_0 * (a**2)) + (b**2) + ((E_0**2) * (c**2))))

def least_squares_fit(sample_estimate_values):
    """Uses least squares to fit the mean_func and sigma_func to the sample
    mean and standard deviation data calculated from the raw data.

    Inputs
    df: pandas dataframe of all detector measurements.
    Returns
    results: dictionary of parameter results"""

    #unpacking values
    mu_samp, mu_error, sigma_samp, sigma_error = sample_estimate_values

    #fitting mean
    mean_params, mean_cov = curve_fit(mean_func, mu_samp.index.values, mu_samp.values, sigma = mu_error.values, absolute_sigma=True)
    mean_params_error = np.sqrt(np.diag(mean_cov))

    #fitting standard deviation
    p0_sig = [0.5, 1.3, 0.01] #setting sigma param initial values
    sigma_params, sigma_cov = curve_fit(sigma_func, sigma_samp.index.values, sigma_samp.values, sigma=sigma_error.values, p0 = p0_sig, absolute_sigma=True, bounds=(0, np.inf))
    sigma_params_error = np.sqrt(np.diag(sigma_cov))

    #merging into one matrix and vector
    all_params = np.concatenate((mean_params, sigma_params))
    full_cov = np.zeros((5, 5))
    full_cov[0:2, 0:2] = mean_cov
    full_cov[2:5, 2:5] = sigma_cov

    #storing parameter results in a dictionary
    param_results = {
        "lb": (mean_params[0], mean_params_error[0]),
        "dE": (mean_params[1], mean_params_error[1]),
        "a":  (sigma_params[0], sigma_params_error[0]),
        "b":  (sigma_params[1], sigma_params_error[1]),
        "c":  (sigma_params[2], sigma_params_error[2]),
    }

    return param_results, all_params, full_cov

def print_and_save_results(param_results, fit_type, filepath='../results.json'):
    """Prints and saves the results of the calculated parameters to 
    results.json file.
    Inputs
    param_results: dictionary of fit parameter values
    fit_type: name of the fit used to calculate params"""
    
    #printing out fitted values
    print("Fitted parameter values:")
    print(f"λ = {param_results['lb'][0]:.3f} ± {param_results['lb'][1]:.3f}")
    print(f"Δ = {param_results['dE'][0]:.3f} ± {param_results['dE'][1]:.3f}")
    print(f"a = {param_results['a'][0]:.3f} ± {param_results['a'][1]:.3f}")
    print(f"b = {param_results['b'][0]:.3f} ± {param_results['b'][1]:.3f}")
    print(f"c = {param_results['c'][0]:.3f} ± {param_results['c'][1]:.3f} \n")

    ### saving to results.json ###
    #opening
    with open(filepath, 'r') as f:
        data = json.load(f)

    #entering data
    for key, (val, err) in param_results.items():
        data[fit_type]["values"][key] = float(val)
        data[fit_type]["errors"][key] = float(err)

    #saving
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

    print("Saved to 'results.json'")

def calculate_error_bands_by_bootstrap(all_params, full_cov, x_arr, n_boot=1000):
    """Calculating the 1 sigma error bands of the fit using parametric 
    bootstrapping of the fitted parameter values lambda, delta, a, b and c.
    Uses percentiles (16th/84th) to handle non-Gaussian distributions.
    
    Inputs
    all_params: fitted parameters for mean and sigma
    full_cov: covariance matrix for mu and sigma parameters
    x_arr: array of smooth x_values across the E_0 value range
    n_boot: number of bootstrap iterations
    
    Returns
    mean_fit_error_band: tuple of (lower_offset, upper_offset) arrays from median
    sigma_fit_error_band: tuple of (lower_offset, upper_offset) arrays from median
    """

    rng = np.random.default_rng()

    #creating n_boots sets of sample parameters from multivariate normal dist
    boot_params = rng.multivariate_normal(all_params, full_cov, size=n_boot)

    #initialising
    boot_mean_curves = []
    boot_sigma_curves = []

    #parametric bootstrapping
    for i in range(n_boot):
        sample = boot_params[i]

        #extracting i'th value
        lb_i, dE_i = sample[0:2]
        a_i, b_i, c_i = sample[2:5]

        #scaling y values
        y_mean_i = mean_func(x_arr, lb_i, dE_i) - x_arr
        y_sigma_i = sigma_func(x_arr, a_i, b_i, c_i) / x_arr

        boot_mean_curves.append(y_mean_i)
        boot_sigma_curves.append(y_sigma_i)

    #calculating error bands using 16th and 84th percentile
    mean_lower = np.percentile(boot_mean_curves, 16, axis=0)
    mean_upper = np.percentile(boot_mean_curves, 84, axis=0)
    mean_median = np.median(boot_mean_curves, axis=0)
    
    sigma_lower = np.percentile(boot_sigma_curves, 16, axis=0)
    sigma_upper = np.percentile(boot_sigma_curves, 84, axis=0)
    sigma_median = np.median(boot_sigma_curves, axis=0)
    
    #tuples of offset sigmas
    mean_fit_error_band = (mean_median - mean_lower, mean_upper - mean_median)
    sigma_fit_error_band = (sigma_median - sigma_lower, sigma_upper - sigma_median)

    return mean_fit_error_band, sigma_fit_error_band

def plot_mean_sigma_fit_with_error_bars(param_results, x_arr, mean_fit_error_band, sigma_fit_error_band, figure, sample_estimate_values=None):
    """Plots two graphs for the mu and sigma values, eaching showing the actual
    mean and std. dev from E_0, as well the line of best fit, and associated error
    bands for that fit.
    
    Inputs
    sample_estimate_values: tuple in form of (mu_samp, mu_error, sigma_samp, sigma_error)
    param_results: dictionary of fit parameter values
    x_arr: array of smooth x_values across the E_0 value range
    mean_fit_error_band: array of uncertainty values for the mean fit at each E_0 value
    sigma_fit_error_band: array of uncertainty values for the sigma fit at each E_0 value
    
    Returns
    fig: the matplotlib figure object, for purposes of saving the image
    """
    
    #creating y values for fitted curves from param results
    fitted_mean = (mean_func(x_arr, param_results['lb'][0], param_results['dE'][0]) - x_arr)
    fitted_sigma = (sigma_func(x_arr, param_results['a'][0], param_results['b'][0], param_results['c'][0]) / x_arr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    #plotting sample estimates of mu and sigma if passed
    if sample_estimate_values != None:
        
        #unpacking values
        mu_samp, mu_error, sigma_samp, sigma_error = sample_estimate_values

        #mean plot
        mu_samp_scaled = mu_samp.values - mu_samp.index.values
        ax1.errorbar(mu_samp.index.values, mu_samp_scaled, fmt='o', label='Data', capsize=4,color='black')#,yerr=mu_error.values)
        #sigma plot
        sigma_samp_scaled = sigma_samp.values / sigma_samp.index.values
        sigma_err_scaled = sigma_error.values / sigma_samp.index.values
        ax2.errorbar(sigma_samp.index.values, sigma_samp_scaled, fmt='o', label='Data', capsize=4,color='black')#, yerr=sigma_err_scaled)
    
    ax1.plot(x_arr, fitted_mean, 'r-', label='Fit')
    mean_lower, mean_upper = mean_fit_error_band
    ax1.fill_between(x_arr, fitted_mean - mean_lower, fitted_mean + mean_upper, color='r', alpha=0.3, label=r'$1\sigma$ Band (percentiles)')
    
    ax1.set_xlabel(r"$E_0$ [GeV]")
    ax1.set_ylabel(r"$\hat{\mu}_{\rm} - E_0$ [GeV]")
    ax1.set_title("Linearity Check")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(x_arr, fitted_sigma, 'r-', label='Fit')
    sigma_lower, sigma_upper = sigma_fit_error_band
    ax2.fill_between(x_arr, fitted_sigma - sigma_lower, fitted_sigma + sigma_upper, color='r', alpha=0.3, label=r'$1\sigma$ Band (percentiles)')

    ax2.set_xlabel(r"$E_0$ [GeV]")
    ax2.set_ylabel(r"$\hat{\sigma}_{\rm} / E_0$")
    ax2.set_title("Fractional Resolution")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.suptitle(figure, y=0.93)
    
    fig.tight_layout()

    return fig

def least_squares_fit_and_plot(sample_estimate_values, fit_type, figure):
    """Applies a least squares to fit of the associated functions to the sample 
    estimate values. Then applies bootstrapping to calculate error bands for the 
    fit at each E_0 value. Plots all of this information onto two graphs, one for
    mu and one for sigma.
    
    Inputs
    sample_estimate_values: tuple in form of (mu_samp, mu_error, sigma_samp, sigma_error)
    fit_type: name of the fit used to calculate params

    Returns
    fig: the matplotlib figure object, for purposes of saving the image
    """

    #applying least squares fit to functions 
    param_results, all_params, full_cov = least_squares_fit(sample_estimate_values)

    #saving parameter results
    print_and_save_results(param_results, fit_type, filepath='../results.json')

    #creating x array to sample y values over
    x_arr = np.linspace(20, 80, 200)

    #applying bootstrapping and saving sigma values
    mean_fit_error_band, sigma_fit_error_band = calculate_error_bands_by_bootstrap(all_params, full_cov, x_arr)

    #plotting
    fig = plot_mean_sigma_fit_with_error_bars(param_results, x_arr, mean_fit_error_band, sigma_fit_error_band, figure, sample_estimate_values=sample_estimate_values)
    
    return fig
