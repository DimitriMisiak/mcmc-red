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

### importing the module functions
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from chi2 import chi2_simple, chi2_freq
from psd import psd

fs = 1e3
t_range = np.arange(0, 1, fs**-1)

# FUNCTION
funk = lambda a: a*t_range

#DATA
sig = 0.02
data = funk(.5) + np.random.normal(0, sig, t_range.shape)
# MODEL
xmod = (.4,.5,.6)
labmod = ('mod1', 'mod2', 'mod3')
darray = {l: funk(a) for l,a in zip(labmod, xmod)}

# TEMPORAL Chi2
d_sx2 = {l: chi2_simple(data, darray[l], err=sig) for l in labmod}

# FREQ Chi2 with fft, psd, etc...
dfft = {l: np.fft.fft(darray[l]) for l in labmod}

fftdata = np.fft.fft(data)
freq, dpsd = psd(fftdata, fs)

noise_list = list()
for k in xrange(100):
    freq, noi = psd(np.fft.fft(np.random.normal(0, sig, t_range.shape)), fs)
    noise_list.append(noi)

npsd = np.mean(noise_list, axis=0)

d_fx2 = {l: chi2_freq(fftdata, dfft[l], npsd, fs) for l in labmod}

########## PLOT #############
plt.close('all')

### TEMPORAL PLOT
plt.figure()
plt.title('1000 pts _ Temporal Chi2')
plt.plot(
    t_range, data, lw=1.,
    label='data, $\chi^2=${:.2f}'.format(chi2_simple(data, data, err=sig))
)

for l in labmod:
    plt.plot(t_range, darray[l], label=l + ' $\chi^2=${:.2f}'.format(d_sx2[l]))
plt.grid(b=True)
plt.legend()

# FREQUENCY PLOT
plt.figure()
plt.title('500 freqs _ Frequency Chi2')
plt.grid(b=True)
plt.loglog(
    freq, dpsd,
    label='data $\chi^2=${:.2f}'.format(chi2_freq(fftdata, fftdata, npsd, fs))
)
for l in labmod:
    freq, PSD = psd(dfft[l], fs)
    plt.loglog(freq, PSD, label=l+' $\chi^2=${:.2f}'.format(d_fx2[l]))
plt.loglog(freq, npsd, label='noise')
plt.legend()
