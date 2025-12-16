"""Main module for s1_sol package"""

import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    """Loading the sample.csv data file"""
    df = pd.read_csv('../sample.csv')
    return df


def set_mpl_style():
    """
    Setting to the S1 mphil style.
    """
    plt.style.use('../s1_sol/mphil.mplstyle')
    print(f"Style applied")