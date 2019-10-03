#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Handy scripting toolbox full of convenient functions.

Author:
    Dimitri Misiak (misiak@ipnl.in2p3.fr)
"""

import matplotlib.pyplot as plt
import scipy.signal as sgl


def explore_plot(x_data, y_data,
                 num='EXPLORE PLOT', figsize=(11,7),
                 label='data', **kwargs):
    """Plots the given x_data and y_data according to the matplot
    keywords given. Also plot the welch psd of the y_data.

    Parameters
    ----------
    x_data : array_like
        Data in abscisse.
    y_data : array_like
        Data in ordinate, must be the same size as x_data.
    num : int or str, optional
        Use to select the figure where to plot the data.
    figsize : tuple of integers, optional, default: (6,4)
        width and height of figure
    Additionnal kwargs will be passed to the plot function.

    Returns
    -------
    figure : Figure

    Examples
    --------
    >>> time_array = np.arange(0, 1, 1e-3)
    >>> funk = lambda t: np.sin(t*13*2*np.pi)
    >>> explore_plot(time_array, funk(time_array),
                     num='test explore_plot')

    """
    fig = plt.figure(num=num, figsize=figsize)
    ax = fig.get_axes()
    # Check if the figure was already plotted.
    if not len(ax):
        fig, ax = plt.subplots(2, num=num, figsize=figsize)
    ax[0].plot(x_data, y_data, label=label, **kwargs)
    ax[0].set_xlabel('Time [$s$]')
    ax[0].set_ylabel('Signal')
    freq, psd = sgl.welch(y_data, fs=1e3, window='boxcar', nperseg=1e3)
    ax[1].plot(freq[1:], psd[1:], label=label, **kwargs)
    ax[1].set_xlabel('Frequency [$Hz$]')
    ax[1].set_ylabel('Signal PSD')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    for a in ax:
        a.legend(fontsize='small', ncol=1)
        a.grid(b=True)
    fig.tight_layout()
