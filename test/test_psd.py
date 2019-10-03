#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:06:16 2018

Test for the function of the chi2 script of the omnitool package.

@author: misiak
"""

import numpy as np
import scipy.signal as sgl
import matplotlib.pyplot as plt

from mcmc_red import psd, angle_psd, inv_psd, psd_from_fft2

plt.close('all')

fs = 1e3
wlen = 2.
N = int(fs*wlen)

time_array = np.arange(0, wlen, fs**-1)

noise_psd = np.ones(N//2) * 1e-5
noise_fun = lambda : inv_psd(noise_psd, fs)

pulse_array = np.exp(-time_array/5e-1)

data_array = pulse_array + noise_fun()

fft_data = np.fft.fft(data_array)

freq_array, psd_array = psd(fft_data, fs)
angle_array = angle_psd(fft_data)

_, psd_welch = sgl.welch(data_array,
                         fs=fs, window='boxcar', nperseg=data_array.size)
psd_welch = psd_welch[1:]

_, psd_fft2 = psd_from_fft2(np.abs(fft_data)**2, fs)

reconstructed_pulse = inv_psd(psd_array, fs,
                              angle=angle_array, mean=data_array.mean())

fig = plt.figure('plot psd test', figsize=(10,10))

axes = fig.subplots(nrows=2, ncols=2)
axes = axes.ravel()

axes[0].plot(time_array, data_array, label='data', lw=3)
axes[0].plot(time_array, reconstructed_pulse, label='reconstructed data', lw=0.5)
axes[0].plot(time_array, pulse_array, label='pulse')
axes[0].plot(time_array, noise_fun(), label='white noise sample')


axes[1].plot(freq_array, noise_psd, label='white noise',
    ls='--', color='r', zorder=10)

axes[1].plot(freq_array, psd_array, label='psd', lw=5)
axes[1].plot(freq_array, psd_welch, label='welch', lw=3)
axes[1].plot(freq_array, psd_fft2, label='psd from fft2', lw=1)

axes[2].plot(freq_array, angle_psd(np.fft.fft(noise_fun())), label='noise sample')
axes[2].plot(freq_array, angle_array, label='data')

axes[3].plot(freq_array, np.abs(psd_array-psd_welch)/psd_array,
    label='residual welch', lw=3)
axes[3].plot(freq_array, np.abs(psd_array-psd_fft2)/psd_array,
    label='residual psd from fft2', lw=3)


axes[1].set_yscale('log')
axes[1].set_xscale('log')

axes[2].set_xscale('log')

axes[3].set_yscale('log')
axes[3].set_xscale('log')

for ax in axes:
    ax.grid()
    ax.legend()
    
fig.tight_layout()

print('Test is done!')
