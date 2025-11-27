"""Question 1 functions"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
import pandas as pd
from scipy.optimize import curve_fit
import os
import matplotlib.cm as cm      
import matplotlib.colors as mcolors

# remove this #
def update_results_json(results, section, filepath='../results.json'):
    """
    Updates a section in the results.json file with the results file passed.
    """
    #opening
    with open(filepath, 'r') as f:
        data = json.load(f)

    #entering data
    for key, (val, err) in results.items():
        data[section]["values"][key] = float(val)
        data[section]["errors"][key] = float(err)

    #saving
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

    print("Saved to 'results.json'")

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
    ax.set_title(r'Distribution of all $E - E_0$')

    return fig, ax

# Question 1) (ii)
def plot_overlapping_hist(df):
    """
    Plotting overlapping histograms of each set of (E - E_0) across each of 
    the E_0 values
    """
    
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    
    #setting histogram plotting colors
    unique_energies = sorted(df['E_true'].unique())
    norm_color = mcolors.Normalize(vmin=min(unique_energies), vmax=max(unique_energies))
    cmap = cm.viridis

    # plotting hist differences across each E_0
    for E_0, group in df.groupby('E_true'):
        diff = group['E_rec'] - E_0
        color = cmap(norm_color(E_0))
        ax.hist(diff, bins=50, histtype='step',color=color, label=f'$E_0={E_0}$')
    
    ax.set_xlabel(r"$(E - E_0)$ [GeV]")
    ax.set_ylabel("Frequency")
    ax.set_title(r'Distribution of $E - E_0$ for each $E_0$')
    ax.legend()
    
    return fig, ax

# Question 1) (iii)
def calculate_sample_estimates(df):
    """
    Calculating the samples estimates and associated estimated 
    errors of the mean and standard deviation.
    """
    
    grouped = df.groupby('E_true')['E_rec']

    #calculating samples values and associated errors
    mu_samp = grouped.mean()
    sigma_samp = grouped.std(ddof=1)
    N = grouped.count()

    mu_error = sigma_samp/ (N ** 0.5)
    sigma_error = sigma_samp/( (2*(N-1)) ** 0.5)

    return (mu_samp, mu_error, sigma_samp, sigma_error)

def plot_sample_estimates(sample_estimate_values):
    """Takes samples estimate values and plots on two separate histograms.
    Returns fig."""

    #unpacking
    mu_samp, mu_error, sigma_samp, sigma_error = sample_estimate_values

    #plotting subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    
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
    sample_estimate_values: tuple in form of (mu_samp, mu_error, sigma_samp, sigma_error)
    Returns
    results: dictionary of parameter results"""

    #unpacking values
    mu_samp, mu_error, sigma_samp, sigma_error = sample_estimate_values

    #fitting mean
    mean_params, mean_params_cov = curve_fit(mean_func, mu_samp.index.values, mu_samp.values, sigma = mu_error.values, absolute_sigma=True)
    mean_params_error = np.sqrt(np.diag(mean_params_cov))

    #fitting standard deviation
    p0_sig = [0.5, 1, 0.01] #setting sigma param initial values
    sigma_params, sigma_params_cov = curve_fit(sigma_func, sigma_samp.index.values, sigma_samp.values, sigma=sigma_error.values, p0 = p0_sig, absolute_sigma=True)
    sigma_params_error = np.sqrt(np.diag(sigma_params_cov))

    #storing parameter results in a dictionary
    param_results = {
        "lb": (mean_params[0], mean_params_error[0]),
        "dE": (mean_params[1], mean_params_error[1]),
        "a":  (sigma_params[0], sigma_params_error[0]),
        "b":  (sigma_params[1], sigma_params_error[1]),
        "c":  (sigma_params[2], sigma_params_error[2]),
    }

    return param_results

def print_and_save_results(param_results, section, filepath='../results.json'):
    """Prints and saves the results of the calculated parameters to 
    results.json file.
    Inputs
    param_results: dictionary of fit parameter values
    section: name of the fit used to calculate params"""
    
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
        data[section]["values"][key] = float(val)
        data[section]["errors"][key] = float(err)

    #saving
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

    print("Saved to 'results.json'")

def calculate_error_bands_by_bootstrap(sample_estimate_values, param_results, x_arr, n_boot=1000):
    """Calculating the 1 sigma error bands of the fit using parametric 
    bootstrapping across the E_0 sample range. y-values are re-scaled 
    to (mu - E_0) and (sigma / E_0)
    
    Inputs
    sample_estimate_values: tuple in form of (mu_samp, mu_error, sigma_samp, sigma_error)
    param_results: dictionary of fit parameter values
    x_arr: array of smooth x_values across the E_0 value range
    
    Returns
    mean_std: array of boostrap derived standard devations of the fit at each E_0 value
    sigma_std: array of boostrap derived standard devations of the fit at each E_0 value 
    """

    #unpacking values
    mu_samp, mu_error, sigma_samp, sigma_error = sample_estimate_values

    #initialising
    boot_mean_curves = []
    boot_sigma_curves = []
    p0_sig = [0.5, 1, 0.01] #setting sigma param initial values

    #parametric bootstrapping
    for i in range(n_boot):
        #creating resamples
        mu_resamp = np.random.normal(mu_samp.values, mu_error.values)
        sigma_resamp = np.random.normal(sigma_samp.values, sigma_error.values)

        #fitting
        boot_mean_params, _ = curve_fit(mean_func, mu_samp.index.values, mu_resamp, sigma = mu_error.values, absolute_sigma=True)
        boot_sigma_params, _ = curve_fit(sigma_func, sigma_samp.index.values, sigma_resamp, sigma=sigma_error.values, p0 = p0_sig, absolute_sigma=True)

        #saving to list
        boot_mean_curves.append(mean_func(x_arr, *boot_mean_params) - x_arr)
        boot_sigma_curves.append(sigma_func(x_arr, *boot_sigma_params) / x_arr)

    #calculating std. deviations across each E_0 for both mean and sigma using boostrap resample curves
    mean_std = np.std(boot_mean_curves,axis=0)
    sigma_std = np.std(boot_sigma_curves,axis=0)

    return mean_std, sigma_std

def plot_mean_sigma_fit_with_error_bars(sample_estimate_values, param_results, x_arr, mean_std, sigma_std):
    """Plots two graphs for the mu and sigma values, eaching showing the actual
    mean and std. dev from E_0, as well the line of best fit, and associated error
    bands for that fit.
    
    Inputs
    sample_estimate_values: tuple in form of (mu_samp, mu_error, sigma_samp, sigma_error)
    param_results: dictionary of fit parameter values
    x_arr: array of smooth x_values across the E_0 value range
    mean_std: array of uncertainty values for the mean fit at each E_0 value
    mean_std: array of uncertainty values for the sigma fit at each E_0 value
    
    Returns
    fig: the matplotlib figure object, for purposes of saving the image
    """
    
    #unpacking values
    mu_samp, mu_error, sigma_samp, sigma_error = sample_estimate_values

    #creating y values for fitted curves from param results
    fitted_mean = (mean_func(x_arr, param_results['lb'][0], param_results['dE'][0]) - x_arr)
    fitted_sigma = (sigma_func(x_arr, param_results['a'][0], param_results['b'][0], param_results['c'][0]) / x_arr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    #mean plot
    mu_samp_scaled = mu_samp.values - mu_samp.index.values
    
    ax1.errorbar(mu_samp.index.values, mu_samp_scaled, yerr=mu_error.values, fmt='o', label='Data', capsize=4,color='black')
    ax1.plot(x_arr, fitted_mean, 'r-', label='Fit')
    ax1.fill_between(x_arr, fitted_mean - mean_std, fitted_mean + mean_std, color='r', alpha=0.3, label=r'$\pm 1\sigma$ Band')
    
    ax1.set_xlabel(r"$E_0$ [GeV]")
    ax1.set_ylabel(r"$\hat{\mu}_{\rm samp} - E_0$ [GeV]")
    ax1.set_title("Linearity Check")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    #sigma plot
    sigma_samp_scaled = sigma_samp.values / sigma_samp.index.values
    sigma_err_scaled = sigma_error.values / sigma_samp.index.values
    
    ax2.errorbar(sigma_samp.index.values, sigma_samp_scaled, yerr=sigma_err_scaled, fmt='o', label='Data', capsize=4,color='black')
    ax2.plot(x_arr, fitted_sigma, 'r-', label='Fit')
    ax2.fill_between(x_arr, fitted_sigma - sigma_std, fitted_sigma + sigma_std, color='r', alpha=0.3, label=r'$\pm 1\sigma$ Band')

    ax2.set_xlabel(r"$E_0$ [GeV]")
    ax2.set_ylabel(r"$\hat{\sigma}_{\rm samp} / E_0$")
    ax2.set_title("Fractional Resolution")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    fig.tight_layout()

    return fig

def least_squares_fit_and_plot(sample_estimate_values):
    """Applies a least squares to fit of the associated functions to the sample 
    estimate values. Then applies bootstrapping to calculate error bands for the 
    fit at each E_0 value. Plots all of this information onto two graphs, one for
    mu and one for sigma.
    
    Inputs
    sample_estimate_values: tuple in form of (mu_samp, mu_error, sigma_samp, sigma_error)
    
    Returns
    fig: the matplotlib figure object, for purposes of saving the image
    """

    #unpacking
    mu_samp, mu_error, sigma_samp, sigma_error = sample_estimate_values

    #applying least squares fit to functions 
    param_results = least_squares_fit(sample_estimate_values)
    print_and_save_results(param_results, "sample_ests", filepath='../results.json')

    #creating x array to sample y values over
    x_arr = np.linspace(min(mu_samp.index.values), max(mu_samp.index.values), 200)

    #appling bootstrapping and saving sigma values
    mean_std, sigma_std = calculate_error_bands_by_bootstrap(sample_estimate_values, param_results, x_arr)

    #plotting
    fig = plot_mean_sigma_fit_with_error_bars(sample_estimate_values, param_results, x_arr, mean_std, sigma_std)
    
    return fig
