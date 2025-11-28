"""Question 4 functions"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from s1_sol import q1, q2, q3

# Question 4) (i)

def run_entire_sample_bootstrap(df, n_boot=2500):
    """
    """
    
    # Dictionary to store lists of parameter results
    boot_results = {
        'sample_ests': {'lb':[], 'dE':[], 'a':[], 'b':[], 'c':[]},
        'individual_fits': {'lb':[], 'dE':[], 'a':[], 'b':[], 'c':[]},
        'simultaneous_fit': {'lb':[], 'dE':[], 'a':[], 'b':[], 'c':[]}
    }
    
    print(f"Running {n_boot} bootstrap iterations...")
    
    for i in range(n_boot):
        
        #resampling the original dataframe
        df_resampled = df.sample(frac=1.0, replace=True)
        
        # 1) samples estimate least squares fit
        ls_vals = q1.calculate_sample_estimates(df_resampled)
        ls_params, _, _ = q1.least_squares_fit(ls_vals)

        #storing results
        for key in boot_results['sample_ests']:
            boot_results['sample_ests'][key].append(ls_params[key][0])
        

        # 2) individual mle fits
        ind_mle_vals = q2.unbinned_mle(df_resampled)
        ind_mle_params, _, _ = q1.least_squares_fit(ind_mle_vals)

        #storing results
        for key in boot_results['individual_fits']:
            boot_results['individual_fits'][key].append(ind_mle_params[key][0])


        # 3) simultaneous mle fits
        sim_mle_params, _, _ = q3.fit_unbinned_mle_simultaneous(df_resampled)
        
        #storing results
        for key in boot_results['simultaneous_fit']:
            boot_results['simultaneous_fit'][key].append(sim_mle_params[key][0])
                
    return boot_results

def plot_bootstrap_histograms(boot_results):
    """
    """
    
    #setting layout and labels
    fig, ax = plt.subplots(2, 3, figsize=(19.2, 9.6))
    
    layout_map = {
        'lb': (0, 0),
        'dE': (0, 1),
        'a':  (1, 0),
        'b':  (1, 1),
        'c':  (1, 2)
    }
    
    x_labels = {
        'lb': r"$\lambda$",
        'dE': r"$\Delta$ [GeV]",
        'a':  r"$a$ [GeV$^{1/2}$]",
        'b':  r"$b$ [GeV]",
        'c':  r"$c$"
    }

    #method styles
    methods = ['sample_ests', 'individual_fits', 'simultaneous_fit']
    method_labels = ['Sample Stats (Q1)', 'Individual MLE (Q2)', 'Simultaneous (Q3)']
    colors = ['black', 'blue', 'red']

    #plotting, loop across each graph
    for param, (row, col) in layout_map.items():
        axis = ax[row, col]
        
        #plotting loop for each method
        for i, method in enumerate(methods):
            if method in boot_results and param in boot_results[method]:
                data = np.array(boot_results[method][param])
                
                #plotting histogram - CHANGED HERE
                # Changed to 'stepfilled' and added alpha for transparency/overlap
                axis.hist(data, bins=50, histtype='stepfilled', density=True, 
                          color=colors[i], label=method_labels[i], alpha=0.4)
                
                # Optional: Add faint vertical line for the mean
                axis.axvline(np.mean(data), color=colors[i], linestyle='--', alpha=0.3)

        #labelling
        axis.set_xlabel(x_labels[param], fontsize=14)
        axis.set_ylabel("Probability Density")
        
        #only adding legen to one plot
        if row == 0 and col == 0:
            axis.legend(loc='upper right', frameon=False)

    #hiding top-right graph
    ax[0, 2].set_visible(False)
    
    fig.tight_layout()
    
    return fig

def boot_and_plot_hists_all_methods(df, n_boot=2500):
    """"""
    boot_results = run_entire_sample_bootstrap(df, n_boot=n_boot)
    fig = plot_bootstrap_histograms(boot_results)

    return fig