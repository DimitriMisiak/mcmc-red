#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:06:16 2018

Test for the function of the chi2 script of the omnitool package.

@author: misiak
"""

import sys
from os import path
import numpy as np
import scipy.signal as sgl
import matplotlib.pyplot as plt

import mcmc_red as mcr

plt.close('all')

def butter_lowpass(cutoff, fs, order=5):
    """
    Design a low-pass filter.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sgl.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fcut, order=5):
    """
    Filter the input data with a low-pass.
    """
    b, a = butter_lowpass(cutoff, fcut, order=order)
    y = sgl.lfilter(b, a, data)
    return y

fs = 1e3
T = 2.
time = np.arange(0, T, fs**-1)

noise = np.random.normal(0, 1.0, size=int(T*fs))
lnoise = butter_lowpass_filter(noise, 10, 1000, order=2)

plt.figure('Temporal')
plt.plot(time, noise, alpha=0.2)
plt.plot(time, lnoise)
plt.grid()

fft = np.fft.fft(lnoise)

freqy, psdy = mcr.psd(fft, fs)

lwelch = np.array(sgl.welch(lnoise, fs, 'boxcar', nperseg=len(noise)))

freq = lwelch[0, 1:]
assert (freq == freqy).all()

plt.figure('PSD')
plt.loglog(*lwelch[:,1:], label='lwelch')
plt.loglog(freq, psdy, ls='--', label='psd')

plt.axhline(np.mean(lwelch[1, 1:-1]), ls='--')
plt.axhline(np.mean(psdy), ls='--')

print(np.mean(lwelch[1, 1:]))
print(np.mean(psdy))
print('Delta =', np.log10(np.mean(lwelch[1, 1:])/np.mean(psdy)))

plt.legend()
plt.grid()
