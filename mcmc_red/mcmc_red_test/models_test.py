#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 20:57:57 2018

Test for the model script of the omnitool package.

@author: misiak
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgl

### importing the omnitool package functions
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from general import explore_plot
from models import model_1exp, model_2exp, model_3exp, dim_psd, dim_ifft, model_1exp_fft

# time array
time = np.arange(0, 1., 1e-3)

# model pulse arrays
mod1 = model_1exp(1, 1e-2, 5e-3, t0=0.1)(time)
mod2 = model_2exp(0.9, 0.1, 1e-2, 1e-1, 5e-3, t0=0.4)(time, details=True)
mod3 = model_3exp(0.7, 0.2, 0.1, 1e-2, 5e-2, 1e-1, 5e-3, t0=0.7)(
    time, details=True
)

# plotting pulses
plt.close('all')

explore_plot(time, mod1, label='mod1')

for data, lab, ls in zip(mod2, ('mod2', 'mod2_exp1', 'mod2_exp2'),
                         ('-','--', '--')):
    explore_plot(time, data, label=lab, ls=ls)

for data, lab, ls in zip(mod3,
                         ('mod3', 'mod3_exp1', 'mod3_exp2', 'mod3_exp3'),
                         ('-','-.', '-.', '-.')):
    explore_plot(time, data, label=lab, ls=ls)


### FFT TEST
tw = 1.
fs = 1e3
fnyq = fs/2
ns = int(tw * fs)
time = np.arange(0, tw, fs**-1)
freq = np.arange(fnyq, 0., -tw**-1)
freq = np.flip(freq, axis=0)

t_step = time[1] - time[0]
t_len = time[-1] - time[0]

param = (1, 6e-3, 5e-3)

fig, ax = plt.subplots(4, figsize=(11,7))

mod_lab = 'mod'
mod1 = model_1exp(*param, t0=0.1)(time)
#    mod1 = np.exp(1j*2*np.pi*3*time)
mod_fft = np.fft.fft(mod1)[1:len(freq)+1]
#    mod_psd = psd_corr * np.abs(mod_fft)**2
mod_psd = dim_psd(mod_fft, t_step, t_len)
mod_welch = sgl.welch(mod1, fs, window='boxcar', nperseg=ns)[1][1:]
mod_angle = np.angle(mod_fft)

noise_lab = 'noise'
noise = np.random.normal(0., 1e-3, size=ns)
noise_fft = np.fft.fft(noise)[1:len(freq)+1]
noise_psd = dim_psd(noise_fft, t_step, t_len)
noise_welch = sgl.welch(noise, fs, window='boxcar', nperseg=ns)[1][1:]
noise_angle = np.angle(noise_fft)

tru_lab = 'true fft'
tru_fft = model_1exp_fft(ns, *param, t0=0.1)(freq)
tru_psd = dim_psd(tru_fft, t_step, t_len)
#    tru_pulse = np.fft.ifft(model_1exp_fft(*param, t0=0.1)(np.fft.fftfreq(ns, fs**-1)) * ns)
tru_pulse = dim_ifft(model_1exp_fft(ns, *param, t0=0.1), fs, ns)
tru_angle = np.angle(tru_fft)

ax[0].plot(time, mod1, label=mod_lab)
ax[1].plot(freq, np.real(mod_fft), label=mod_lab)
ax[2].plot(freq, np.imag(mod_fft), label=mod_lab)
ax[3].plot(freq, mod_psd, label=mod_lab)

ax[0].plot(time, noise, label=noise_lab)
#    ax[1].plot(freq, -np.real(noise_fft), label=noise_lab)
#    ax[2].plot(freq, -np.imag(noise_fft), label=noise_lab)
ax[3].plot(freq, noise_psd, label=noise_lab)

ax[0].plot(time, tru_pulse, label=tru_lab, lw=1.)
ax[1].plot(freq, np.real(tru_fft), label=tru_lab, lw=1.)
ax[2].plot(freq, np.imag(tru_fft), label=tru_lab, lw=1.)
ax[3].plot(freq, tru_psd, label=tru_lab)

ax[3].set_xscale('log')
ax[3].set_yscale('log')
#    ax[2].set_yscale('log')
#    ax[1].set_yscale('log')

ax[3].plot(freq, mod_welch, label='welch')
#    ax[3].plot(freq, noise_welch, label='noise welch')

for a in ax:
    a.legend()
    a.grid()
fig.tight_layout()
