"""Question 3 functions"""

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
import pandas as pd
import json


# Question 3) (i)

def mu_sig_func(E_true, lam, delta, a, b, c):
    """Calculates the mu and sigma values of the total data set given these associated parameters."""
    
    mu = lam * E_true + delta
    sigma = np.sqrt(np.abs((a**2 * E_true) + b**2 + (c * E_true)**2))

    return mu, sigma

def fit_and_plot_simultaneous(df, n_boot=1000):
    """Performs a simultaneous unbinned likelihood fit over all the data points and then plots the results"""

    #placing data into numpy arrays
    E_rec = df['E_rec'].values
    E_true = df['E_true'].values

    ### defining a cost function ###
    def total_nll(lam, delta, a, b, c):

        #calculate predictions for each data point
        mu_vec, sigma_vec = mu_sig_func(E_true, lam, delta, a, b, c)

        #calculate z-score
        z_score = (E_rec - mu_vec) / sigma_vec

        #calculate nll, ignore log(2pi) term as it doesnt affect the minimum
        nll = np.sum(np.log(sigma_vec) + (0.5 * z_score**2))

        return nll

    ### applying minimisation with Minuit ###
    #using initial guesses from Q1
    m = Minuit(total_nll, lam = 1.0, delta = 2.0, a = 0.5, b = 1.0, c = 0.05)

    #running fits
    m.migrad()
    m.hesse()

    #extracting and saving values
    values = m.values
    errors = m.errors
    best_params = [values['lam'], values['delta'], values['a'], values['b'], values['c']]


    #printing out fitted values
    print("Fitted parameter values:")
    print(f"λ = {values['lam']:.3f} ± {errors['lam']:.3f}")
    print(f"Δ = {values['delta']:.3f} ± {errors['delta']:.3f}")
    print(f"a = {values['a']:.3f} ± {errors['a']:.3f}")
    print(f"b = {values['b']:.3f} ± {errors['b']:.3f}")
    print(f"c = {values['c']:.3f} ± {errors['c']:.3f} \n")

    ### calculating error bands by bootstrapping ###
    x_arr = np.linspace(min(E_true), max(E_true), 100)
    boot_mean_curves = []
    boot_sigma_curves = []

    #bootstrapping
    for i in range(n_boot):
        
        #creating resamples
        mu_sim, sigma_sim = mu_sig_func(E_true, *best_params)
        E_rec_sim = np.random.normal(mu_sim, sigma_sim)
        
        #defining loss function again but using bootstrapping variable E_rec_sim instead
        def boot_nll(lam, delta, a, b, c):
            mu_v, sig_v = mu_sig_func(E_true, lam, delta, a, b, c)
            z = (E_rec_sim - mu_v) / sig_v
            return np.sum(np.log(sig_v) + 0.5 * z**2)
            
        #fitting
        m_boot = Minuit(boot_nll, lam=values['lam'], delta=values['delta'], 
                        a=values['a'], b=values['b'], c=values['c'])
        m_boot.errordef = Minuit.LIKELIHOOD
        m_boot.migrad()
        v = m_boot.values
        
        mu_curve, sigma_curve = mu_sig_func(x_arr, v['lam'], v['delta'], v['a'], v['b'], v['c'])

        boot_mean_curves.append(mu_curve - x_arr)
        boot_sigma_curves.append(sigma_curve / x_arr)

    #calculating error bands
    mean_std = np.std(boot_mean_curves, axis=0)
    sigma_std = np.std(boot_sigma_curves, axis=0)


    ### plotting ###

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    
    #creating y values for fitted curves
    mu_fit, sigma_fit = mu_sig_func(x_arr, *best_params)
    fitted_mean = mu_fit - x_arr
    fitted_sigma = sigma_fit / x_arr
    
    #calculating sample stats
    grouped = df.groupby('E_true')['E_rec']
    mu_samp = grouped.mean()
    sigma_samp = grouped.std(ddof=1)
    N = grouped.count()
    
    #sample errors
    mu_error = sigma_samp / (N ** 0.5)
    sigma_error = sigma_samp / ((2 * (N - 1)) ** 0.5)
    
    #scaling y values
    mu_samp_scaled = mu_samp.values - mu_samp.index.values
    sigma_samp_scaled = sigma_samp.values / sigma_samp.index.values
    sigma_err_scaled = sigma_error.values / sigma_samp.index.values
    
    #plotting left-hand graph
    ax1.errorbar(mu_samp.index.values, mu_samp_scaled, yerr=mu_error.values, fmt='o', label='Sample Estimates', capsize=4, color='black')
    ax1.plot(x_arr, fitted_mean, 'r-', label='Simultaneous Fit')
    ax1.fill_between(x_arr, fitted_mean - mean_std, fitted_mean + mean_std, 
                     color='r', alpha=0.3, label=r'$\pm 1\sigma$ Band')
    
    ax1.set_xlabel(r"$E_0$ [GeV]")
    ax1.set_ylabel(r"$\hat{\mu} - E_0$ [GeV]")
    ax1.set_title("Linearity (Simultaneous Fit)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    #plotting righ-hand graph
    ax2.errorbar(sigma_samp.index.values, sigma_samp_scaled, yerr=sigma_err_scaled, fmt='o', label='Sample Estimates', capsize=4, color='black')
    ax2.plot(x_arr, fitted_sigma, 'r-', label='Simultaneous Fit')
    ax2.fill_between(x_arr, fitted_sigma - sigma_std, fitted_sigma + sigma_std, 
                     color='r', alpha=0.3, label=r'$\pm 1\sigma$ Band')
                     
    ax2.set_xlabel(r"$E_0$ [GeV]")
    ax2.set_ylabel(r"$\hat{\sigma} / E_0$")
    ax2.set_title("Resolution (Simultaneous Fit)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()

    #saving and returning results in a dictionary
    results = {
        "lb": (values['lam'], errors['lam']),
        "dE": (values['delta'], errors['delta']),
        "a":  (values['a'], errors['a']),
        "b":  (values['b'], errors['b']),
        "c":  (values['c'], errors['c']),
    }
    
    return results, fig

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