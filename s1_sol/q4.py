"""Question 4 functions"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from s1_sol import q1, q2, q3
from matplotlib.lines import Line2D

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
    method_labels = ['Sample Estimate', 'Individual Fit', 'Simultaneous Fit']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    #plotting, loop across each graph
    for param, (row, col) in layout_map.items():
        axis = ax[row, col]
        
        #creating array for commons bins each parameter
        all_data_for_param = []
        for method in methods:
            all_data_for_param.extend(boot_results[method][param])
        common_bins = np.linspace(min(all_data_for_param), max(all_data_for_param), 50)

        #plotting loop for each method
        for i, method in enumerate(methods):
            #if method in boot_results and param in boot_results[method]:
            data = np.array(boot_results[method][param])
            
            #plotting histograms
            axis.hist(data, bins=common_bins, histtype='step', density=True, color=colors[i], label=method_labels[i], linewidth=2)

        #labelling
        axis.set_xlabel(x_labels[param], fontsize=14)
        axis.set_ylabel("Probability Density")
        
        #only adding legend to one plot
        if row == 0 and col == 0:
            axis.legend(loc='upper right', frameon=False)

    #hiding top-right graph
    ax[0, 2].set_visible(False)
    
    fig.tight_layout()
    plt.suptitle("Figure 4.1: Non-parametric Boostraps Parameter Histograms by Method", y=1.03)
    
    return fig

def boot_and_plot_hists_all_methods(df, n_boot=2500):
    """"""
    boot_results = run_entire_sample_bootstrap(df, n_boot=n_boot)
    fig = plot_bootstrap_histograms(boot_results)

    return fig, boot_results

# Question 4) (ii)

def calculate_bootstrap_stats(boot_results):
    """
    Calculates the mean and standard deviation for each parameter and method
    from the raw bootstrap samples.
    """
    stats = {}
    
    # Loop through methods
    for method, params_dict in boot_results.items():
        # Initialize the structure the plotter expects
        stats[method] = {'values': {}, 'errors': {}}
        
        # Loop through parameters
        for param, values in params_dict.items():
            # Separate values and errors into their own dictionaries
            stats[method]['values'][param] = np.mean(values)
            stats[method]['errors'][param] = np.std(values)
                
    return stats

def plot_overlay_comparison(original_data, boot_data):
    """
    Plots original results vs bootstrap results on the same axes.
    Original Data = Solid Markers
    Bootstrap Data = Hollow Markers
    """

    #setting up labels and namings
    params = ['lb', 'dE', 'a', 'b', 'c']
    param_labels = [r'$\lambda$', r'$\Delta$', r'$a$', r'$b$', r'$c$']
    methods = ['sample_ests', 'individual_fits', 'simultaneous_fit']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    method_names = ['Sample Estimate', 'Individual Fit', 'Simultaneous Fit']

    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    #creating insets
    ax_ins_lb = ax.inset_axes([0.14, 0.53, 0.15, 0.12])
    ax_ins_c = ax.inset_axes([0.83, 0.09, 0.15, 0.12])

    #looping over each parameter and plotting
    for i, param in enumerate(params):
        
        #loop over each method
        for j, method in enumerate(methods):
            
            #extract original data
            val_orig = original_data[method]['values'][param]
            err_orig = original_data[method]['errors'][param]
            
            #extract bootstrap data
            val_boot = boot_data[method]['values'][param]
            err_boot = boot_data[method]['errors'][param]
            
            #offsetting points
            center_offset = (j - 1) * 0.25
            x_orig = i + center_offset - 0.05
            x_boot = i + center_offset + 0.05
            
            #plotting legend only once
            label_orig = f"{method_names[j]} (Orig)" if i == 0 else None
            label_boot = f"{method_names[j]} (Boot)" if i == 0 else None
            
            #plotting
            ax.errorbar(x_orig, val_orig, yerr=err_orig, fmt='o', color=colors[j], alpha=1, capsize=3, label=label_orig)
            ax.errorbar(x_boot, val_boot, yerr=err_boot, fmt='o', color=colors[j], mfc='white', mew=1.5, capsize=3, label=label_boot)

            #adding inset plots
            if param == 'lb':
                #plotting original
                ax_ins_lb.errorbar(x_orig, val_orig, yerr=err_orig, fmt='o', color=colors[j], capsize=3)
                #plotting bootstrap
                ax_ins_lb.errorbar(x_boot, val_boot, yerr=err_boot, fmt='o', color=colors[j], mfc='white', mew=1.5, capsize=3)
            
            elif param == 'c':
                #plotting original
                ax_ins_c.errorbar(x_orig, val_orig, yerr=err_orig, fmt='o', color=colors[j], capsize=3)
                #plotting bootstrap
                ax_ins_c.errorbar(x_boot, val_boot, yerr=err_boot, fmt='o', color=colors[j], mfc='white', mew=1.5, capsize=3)

    #set axes labels
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(param_labels, fontsize=12)
    ax.set_ylabel("Parameter Value")
    ax.legend(ncol=1, fontsize='small', loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_title("Fig 4.2: Comparison: Analytical vs Bootstrap Errors", pad=10)

    #lambda inset
    ax_ins_lb.set_ylim(1.0094, 1.0145)
    ax_ins_lb.set_xlim(-0.4, 0.4)
    ax_ins_lb.set_xticks([])
    ax_ins_lb.grid(axis='y', linestyle='--', alpha=0.5)
    ax_ins_lb.set_title(r'zoomed $\lambda$', fontsize=10)

    #c inset
    ax_ins_c.set_ylim(0.022, 0.045) 
    ax_ins_c.set_xlim(3.6, 4.4)
    ax_ins_c.set_xticks([])
    ax_ins_c.grid(axis='y', linestyle='--', alpha=0.5)
    ax_ins_c.set_title(r'zoomed $c$', fontsize=10)
    
    return fig