"""Question 4 functions"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample
from scipy.optimize import curve_fit
from scipy.stats import norm
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
import json
import os

