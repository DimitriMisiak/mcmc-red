#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testing for the ptmcmc script.

Author:
    Dimitri Misiak (misiak@ipnl.in2p3.fr)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgl
import sys
from os import path

### importing the omnitool package functions
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from general import explore_plot
from ptmcmc import (get_ptmcmc_sampler, ptmcmc_results, ptmcmc_sampler,
                    chi2, save_ptmcmc_sampler, ptmcmc_plots)
import models as md


# close all plots
plt.close('all')

# XXX Time array. Need the window length and the sampling frequency
# creating time array
time_array = np.arange(0, 1, 1e-3)

# model function to study
#funk = lambda p: md.model_1exp(*p, t0=0.5)
funk = lambda p: md.model_2exp(*p, t0=0.5)

# XXX Creating the fake data.
# Need to noise level, and the systematic error
# creating noise
noise = 1e-6 * (np.random.rand(1,len(time_array))[0] - 0.5)

# creating experimental data
#    pc = (100, 1e-2, 5e-3)
#    DATA = funk(pc) + noise
pc = (1e-4, 1e-5, 1e-2, 1e-1, 5e-3)
neat_pulse = funk(pc)(time_array, details=True)
DATA = neat_pulse[0] + noise

# plotting experimental data
explore_plot(time_array, DATA, label='Exp. Pulse\n{}'.format(pc),
             num='test explore_plot')
from itertools import count
gen = count()
for exp in neat_pulse[1:]:
    explore_plot(time_array, exp, label='exp '+str(gen.next()),
                 num='test explore_plot')

# defining the chi2 function from the noise psd
freq, psd = sgl.welch(noise, fs=1e3, window='boxcar', nperseg=1000)
psd = psd[1:]
def aux(f):
    f = np.power(10, f)
    x2 = chi2(time_array, funk(f)(time_array), DATA, noise=psd)[0]
    return x2

# save directory
sampler_path = 'ptmcmc_sampler/autosave_2'

# running the mcmc analysis
#    bounds = ((0., 4.), (-6, 0.), (-6, 0.))
#    bounds = ((-4.5, 3.5), (-5.5, -4.5), (-2.5, -1.5), (-1.5, -0.5), (-3, -2))
bounds = ((-5.5, -3.5), (-5.5, -3.5), (-2.5, -0.5), (-2.5, -0.5), (-3, -2))
#    bounds = ((-6, 0), (-6, 0), (-6, 0), (-6, 0), (-6, 0))
condi = lambda p: p[2] < p[3]
sampler = ptmcmc_sampler(aux, bounds, nsteps=2000, ntemps=5, condi=condi)
#
#    # save sampler data
save_ptmcmc_sampler(sampler, bounds, path = sampler_path)


# XXX analysing the ptmcmc results
#    # loading the mcmc results
logd, chain, lnprob, acc = get_ptmcmc_sampler(sampler_path)
#
#    LAB = ('$log\ a$', '$log\ t$', '$log\ s$')
LAB = ('$log\ a1$', '$log\ a2$', '$log\ t1$', '$log\ t2$', '$log\ s$')
dim = int(logd['dim'])
ntemps = int(logd['ntemps'])

ptmcmc_plots(ntemps, dim, chain, lnprob, acc, LAB)

xopt, inf, sup = ptmcmc_results(dim, chain, lnprob, acc, LAB)

popt, pinf, psup = map(lambda x: np.power(10,x),
                       (xopt, inf, sup))

MOD = funk(popt)(time_array)
MOD_lab = 'param= {}'.format(popt)

explore_plot(time_array, MOD,
             num='test explore_plot', label=MOD_lab)
