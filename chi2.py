#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scripts gathering different chi2 functions.

Author:
    Dimitri Misiak (misiak@ipnl.in2p3.fr)
"""

import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

from .psd import psd

def chi2_simple(data_1, data_2, err):
    """ Simplest chi2 function comparing two sets of datas.

    Parameters
    ==========
    data_1, data_2 : array_like
        Set of data to be compared.
    err : float or array_like
        Error affecting the data. If float is given, the error is the same
        for each points. If array_like, each point is affected by its
        corresponding error.

    Returns
    =======
    x2 : float
        Chi2 value.
    """
    array_1 = np.array(data_1)
    array_2 = np.array(data_2)
    err_array = np.array(err)
    x2 = np.sum( (array_1 - array_2)**2  / err_array**2 )
    return x2


def chi2_freq(fft_1, fft_2, psd_err, fs):
    """ Chi2 function comparing two sets of datas through their FFT, taking as
    reference the the PSD of the noise level.

    Parameters
    ==========
    fft_1, fft_2 : array_like
        FFT of data to be compared. Their frequency are ordered as
        the numpy.fft.fft function result (i.e. [0, positive, negative]).
    psd_err : float or array_like
        PSD of the error affecting the data. If float is given, the error is the same
        for each points. If array_like, each point is affected by its
        corresponding error.
    fs : float or int
        Sampling frequency.

    Returns
    =======
    x2 : float
        Chi2 value.
    """
    assert len(fft_1) == len(fft_2)
    assert len(psd_err) == 1 or len(psd_err) == len(fft_1)/2

    psd_data = psd(fft_1 - fft_2, fs)[1]

    x2 = np.sum( psd_data / np.array(psd_err) )

    return x2


def opt_chi2_freq(fft_data, model_funk, psd_err, fs, bounds, debug=False):
    """

    Return:
    =======
    x2 : float
        Chi2 value with the optimal free parameter value.
    xopt : float
        Optimal free parameter value.
    """
    def aux(x):
        fft_model = np.fft.fft(model_funk(x))
        x2 = chi2_freq(fft_data, fft_model, psd_err, fs)
        return x2

    x0 = np.mean(bounds)
    result = op.minimize(aux, x0, method='nelder-mead')
    x2 = result.fun
    xopt = result.x[0]

    if debug == True:

        comm = np.linspace(bounds[0], bounds[1], 100)
#        loot = map(aux, comm)
        loot = [aux(c) for c in comm]
        
        xmin = comm[np.argmin(loot)]

        plt.figure('debug stream_chi2')
        scatter = plt.plot(comm, loot, ls='none', marker='+',
                           label='{0:.3f}/{1:.6f}'.format(xmin, xopt))
        plt.axvline(xmin, lw=1.0, color=scatter[0].get_color())
        plt.ylabel('$\chi^2$')
        plt.xlabel('Model Offset $x$')
        plt.grid(b=True)
        plt.legend(title='Crude/Mini')
        plt.tight_layout()

    return x2, xopt
