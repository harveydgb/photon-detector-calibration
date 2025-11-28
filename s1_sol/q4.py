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
        
        # 2) individual mle fits
        ind_mle_vals = q2.unbinned_mle(df_resampled)
        ind_mle_params, _, _ = q1.least_squares_fit(ind_mle_vals)

        # 3) simultaneous mle fits

        # Store LS results
        for key in boot_results['sample_ests']:
            boot_results['sample_ests'][key].append(ls_res[key][0]) # [0] is value, [1] is error

        # --- Method 2: Q3 Simultaneous MLE ---
        # 1. Direct fit on resampled data
        sim_res, _, _ = fit_unbinned_mle_simultaneous(df_resampled)
        
        # Store Sim results
        for key in boot_results['simultaneous_fit']:
            boot_results['simultaneous_fit'][key].append(sim_res[key][0])
                

    return boot_results