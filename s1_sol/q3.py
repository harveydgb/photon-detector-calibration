"""Question 3 functions"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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

        #prevents sigma from being zero-valued
        sigma_vec = np.maximum(sigma_vec, 1e-9)
        
        #calculate z-score
        z_score = (E_rec - mu_vec) / sigma_vec

        #calculate nll, ignore log(2pi) term as it doesnt affect the minimum
        nll = np.sum(np.log(sigma_vec) + (0.5 * z_score**2))

        return nll

    ### applying minimisation with Minuit ###
    #using initial guesses from Q1
    m = Minuit(total_nll, lb = 1.0, dE = 2.0, a = 0.5, b = 1.3, c = 0.05)

    #setting limits for a, b and c to stay positive
    m.limits["a"] = (0, None)
    m.limits["b"] = (0, None)
    m.limits["c"] = (0, None)


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

def fit_and_plot_simultaneous_unbinned_mle(df, fit_type, figure):
    
    #applying simultaneous mle fit
    param_results, all_params, full_cov = fit_unbinned_mle_simultaneous(df)

    #saving parameter results
    q1.print_and_save_results(param_results, fit_type, filepath='../results.json')

    #creating x array to sample y values over
    x_arr = np.linspace(20, 80, 200)

    #appling bootstrapping and saving sigma values
    mean_fit_error_band, sigma_fit_error_band = q1.calculate_error_bands_by_bootstrap(all_params, full_cov, x_arr)

    #plotting
    fig = q1.plot_mean_sigma_fit_with_error_bars(param_results, x_arr, mean_fit_error_band, sigma_fit_error_band, figure)

    return fig

# Question 3) (iii)

def load_data(filepath='../results.json'):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def plot_parameter_comparison(data):
    #setting up labels and namings
    params = ['lb', 'dE', 'a', 'b', 'c']
    param_labels = [r'$\lambda$', r'$\Delta$', r'$a$', r'$b$', r'$c$']
    methods = ['sample_ests', 'individual_fits', 'simultaneous_fit']
    method_labels = ['Sample Estimate', 'Individual Fit', 'Simultaneous Fit']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    plt.suptitle("Figure 3.2: Parameter Values by Method")

    
    #creating insets
    ax_ins_lb = ax.inset_axes([0.14, 0.55, 0.12, 0.12])
    ax_ins_c = ax.inset_axes([0.86, 0.12, 0.12, 0.12])

    #looping over each parameter and plotting
    for i, param in enumerate(params):
        for j, method in enumerate(methods):
            val = data[method]['values'][param]
            err = data[method]['errors'][param]
            offset = (j - 1) * 0.2
            x_position = i + offset
            
            #plotting legend only once
            label = method_labels[j] if i == 0 else None
            
            ax.errorbar(x_position, val, yerr=err, fmt='o', color=colors[j], capsize=4, label=label)

            #adding inset plots
            if param == 'lb':
                ax_ins_lb.errorbar(x_position, val, yerr=err, fmt='o', color=colors[j], capsize=4)
            elif param == 'c':
                ax_ins_c.errorbar(x_position, val, yerr=err, fmt='o', color=colors[j], capsize=4)

    #set axes labels
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(param_labels, fontsize=12)
    ax.set_ylabel(r"Parameter Value", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.1, 2.2)

    #lambda inset
    ax_ins_lb.set_ylim(1.009, 1.015) 
    ax_ins_lb.set_xlim(-0.3, 0.3)
    ax_ins_lb.set_xticks([])
    ax_ins_lb.grid(axis='y', linestyle='--', alpha=0.5)
    ax_ins_lb.set_title(r'zoomed $\lambda$', fontsize=10)

    #c inset
    ax_ins_c.set_ylim(0.022, 0.045)
    ax_ins_c.set_xlim(3.7, 4.3)
    ax_ins_c.set_xticks([])
    ax_ins_c.grid(axis='y', linestyle='--', alpha=0.5)
    ax_ins_c.set_title(r'zoomed $c$', fontsize=10)
    
    return fig