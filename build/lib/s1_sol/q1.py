"""Question 1 functions"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt
import json
import requests



#Question 1) (i)
def plot_total_hist(df):
    """
    Plotting Histogram of all E - E_0 values.
    """
    
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    
    # Calculate E - E_0
    diff = df['E_true'] - df['E_rec']
    
    # Plot histogram
    ax.hist(diff, bins=50)
    ax.set_xlabel('E - E_0')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of E - E_0')
    ax.show()
    return fig, ax