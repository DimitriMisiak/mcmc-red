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
import matplotlib.pyplot as plt

import mcmc_red as mcr

plt.close('all')

fs = 1e3
t_range = np.arange(0, 1, fs**-1)

# FUNCTION
funk = lambda t,a: np.heaviside(t_range-t, 1.) * a * (np.exp(t_range-t)-1)

# DATA
sig = 0.02
data = funk(.5, 1.) + np.random.normal(0, sig, t_range.shape)

# MODEL
tmod = (.4,.5,.6)
xmod = (0.5, 1., 1.5)
labmod = ('mod1', 'mod2', 'mod3')
darray = {l: funk(t,a) for l,t,a in zip(labmod, tmod, xmod)}

# TEMPORAL Chi2
d_sx2 = {l: mcr.chi2_simple(data, darray[l], err=sig) for l in labmod}

# FREQ Chi2 with fft, psd, etc...
dfft = {l: np.fft.fft(darray[l]) for l in labmod}

fftdata = np.fft.fft(data)
freq, dpsd = mcr.psd(fftdata, fs)

noise_list = list()
for k in range(100):
    freq, noi = mcr.psd(np.fft.fft(np.random.normal(0, sig, t_range.shape)), fs)
    noise_list.append(noi)
npsd = np.mean(noise_list, axis=0)

d_fx2 = {l: mcr.chi2_freq(fftdata, dfft[l], npsd, fs) for l in labmod}

# OPT Chi2 with free parameter

opt_funk = lambda t: funk(t, 1.3)
bounds = (0., 1)
opt_x2, opt_t = mcr.opt_chi2_freq(fftdata, opt_funk, npsd, fs, bounds, debug=True)
opt_mod = opt_funk(opt_t)

########## PLOT #############

### TEMPORAL PLOT
plt.figure()
plt.title('1000 pts _ Temporal Chi2')
plt.plot(
    t_range, data, lw=1.,
    label='data, $\chi^2=${:.2f}'.format(mcr.chi2_simple(data, data, err=sig))
)

for l in labmod:
    plt.plot(t_range, darray[l], label=l + ' $\chi^2=${:.2f}'.format(d_sx2[l]))

plt.plot(t_range, opt_mod, ls='--', color='red', label='OPT')
plt.grid(b=True)
plt.legend()

# FREQUENCY PLOT
plt.figure()
plt.title('500 freqs _ Frequency Chi2')
plt.grid(b=True)
plt.loglog(
    freq, dpsd,
    label='data $\chi^2=${:.2f}'.format(mcr.chi2_freq(fftdata, fftdata, npsd, fs))
)
for l in labmod:
    freq, PSD = mcr.psd(dfft[l], fs)
    plt.loglog(freq, PSD, label=l+' $\chi^2=${:.2f}'.format(d_fx2[l]))
plt.loglog(freq, npsd, label='noise')
plt.legend()
