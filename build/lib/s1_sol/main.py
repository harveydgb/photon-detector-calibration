"""Main module for s1_sol package"""

from pathlib import Path
import pandas as pd


def register_style():
    """Register the mphil matplotlib style with matplotlib"""
    import matplotlib.style.core as mstyle
    
    style_dir = Path(__file__).parent
    style_file = style_dir / "mphil.mplstyle"
    
    if style_file.exists():
        if str(style_dir) not in mstyle.USER_LIBRARY_PATHS:
            mstyle.USER_LIBRARY_PATHS.append(str(style_dir))


def load_data():
    """Loading the sample.csv data file"""
    print('test2')
    df = pd.read_csv('../sample.csv')
    return df
