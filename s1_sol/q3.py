"""Question 3 functions"""

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
import pandas as pd
import json
from s1_sol import q1


# Question 3) (i)

def mu_sig_func(E_true, lb, dE, a, b, c):
    """Calculates the mu and sigma values of the total data set given these associated parameters."""
    
    mu = lb * E_true + dE
    sigma = np.sqrt(np.abs((a**2 * E_true) + b**2 + (c * E_true)**2))

    return mu, sigma

def fit_unbinned_mle_simultaneous(df):
    #placing data into numpy arrays
    E_rec = df['E_rec'].values
    E_true = df['E_true'].values

    ### defining a cost function ###
    def total_nll(lb, dE, a, b, c):

        #calculate predictions for each data point
        mu_vec, sigma_vec = mu_sig_func(E_true, lb, dE, a, b, c)

        #calculate z-score
        z_score = (E_rec - mu_vec) / sigma_vec

        #calculate nll, ignore log(2pi) term as it doesnt affect the minimum
        nll = np.sum(np.log(sigma_vec) + (0.5 * z_score**2))

        return nll

    ### applying minimisation with Minuit ###
    #using initial guesses from Q1
    m = Minuit(total_nll, lb = 1.0, dE = 2.0, a = 0.5, b = 1.0, c = 0.05)
    m.errordef = Minuit.LIKELIHOOD #telling minuit this is a likelihood

    #running fits
    m.migrad()
    m.hesse()

    #getting full covariance matrix
    all_params = np.array(list(m.values))
    full_cov = np.array(m.covariance)

    #extracting and saving values
    values = m.values
    errors = m.errors

    param_results = {
        "lb": (values['lb'], errors['lb']),
        "dE": (values['dE'], errors['dE']),
        "a":  (values['a'], errors['a']),
        "b":  (values['b'], errors['b']),
        "c":  (values['c'], errors['c']),
    }

    return param_results, all_params, full_cov

def fit_and_plot_simultaneous_unbinned_mle(df, fit_type):
    
    #applying simultaneous mle fit
    param_results, all_params, full_cov = fit_unbinned_mle_simultaneous(df)

    #saving parameter results
    q1.print_and_save_results(param_results, fit_type, filepath='../results.json')

    #creating x array to sample y values over
    x_arr = np.linspace(20, 80, 200)

    #appling bootstrapping and saving sigma values
    mean_fit_error_band, sigma_fit_error_band = q1.calculate_error_bands_by_bootstrap(all_params, full_cov, x_arr)

    #plotting
    fig = q1.plot_mean_sigma_fit_with_error_bars(param_results, x_arr, mean_fit_error_band, sigma_fit_error_band)

    return fig

# Question 3) (iii)

def plot_parameter_comparison(filepath='../results.json'):
    """
    Plots a comparison of parameter estimates from the three different methods.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    #setting up labels and namings
    params = ['lb', 'dE', 'a', 'b', 'c']
    latex_labels = [r'$\lambda$', r'$\Delta$ [GeV]', r'$a$ [GeV$^{1/2}$]', r'$b$ [GeV]', r'$c$']
    methods = ['sample_ests', 'individual_fits', 'simultaneous_fit']
    method_names = ['Sample Stats (Q1)', 'Individual MLE (Q2)', 'Simultaneous (Q3)']
    colors = ['black', 'blue', 'red']
    markers = ['o', 's', '^']

    #creating subplots
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), constrained_layout=True)

    #looping over each parameter and plotting
    for i, param in enumerate(params):
        ax = axes[i]
        
        #looping over each method per parameter
        for j, method in enumerate(methods):
            val = data[method]['values'][param]
            err = data[method]['errors'][param]
            
            #plotting errorbars
            ax.errorbar(j, val, yerr=err, fmt=markers[j], color=colors[j], capsize=5, markersize=6, label=method_names[j] if i == 2 else "")

        #formatting subplots
        ax.set_title(latex_labels[i])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(['Q1', 'Q2', 'Q3'])
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')

        #adjusting margins
        ax.set_xlim(-0.5, 2.5)

    #creting a global legend
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)
    
    return fig



