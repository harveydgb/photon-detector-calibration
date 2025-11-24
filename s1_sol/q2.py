"""Question 2 functions"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL


def plot_unbinned_mle(df):
    """Performs an unbinned ML fit for each E_0 using iminuit and plots the inidividual fits of each E_0 overlaid."""


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    #creating x array and an empty y array for the total pdf to fill
    x_arr = np.linspace(-20, 20, 1000)
    total_summed_pdf = np.zeros_like(x_arr)

    fit_results = {}

    #defining function iminuit can read
    def gaussian_pdf(x, mu, sigma):
        return norm.pdf(x, loc=mu, scale=sigma)

    #iterating over each E_0 group
    for E_0, group in df.groupby('E_true'):
        data = group['E_rec'].values

        nll = UnbinnedNLL(data, gaussian_pdf)

        # providing minuit with guesses of sample stats
        m = Minuit(nll, mu=np.mean(data), sigma=np.std(data))

        #running minimisation and error analysis
        m.migrad()
        m.hesse()

        #extracting best fit values
        mu_hat = m.values['mu']
        sigma_hat = m.values['sigma']
        mu_error = m.errors['mu']
        sigma_error = m.errors['sigma']

        fit_results[E_0] = {
            'mu': (mu_hat, mu_error), 
            'sigma': (sigma_hat, sigma_error)
        }

        residuals = data - E_0

        #plotting
        counts, bins, _ = ax1.hist(residuals, bins=50, histtype='step', label=f'$E_0={E_0}$', alpha=0.8)
        bin_width = bins[1] - bins[0]
        n_events = len(data)

        pdf_y = norm.pdf(x_arr, loc=(mu_hat - E_0), scale = sigma_hat)

        #scale the pdf to match the histogram area
        scaled_pdf = pdf_y * (n_events * bin_width)

        #plotting the curve
        ax1.plot(x_arr, scaled_pdf, linestyle='--', linewidth=1, alpha = 0.8)

        total_summed_pdf += scaled_pdf


    return fit_results, fig
    

