#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Handy MCMC scripts.

Test for the different fit method (mcmc, ptmcmc, minimizer).

Author:
    Dimitri Misiak (misiak@ipnl.in2p3.fr)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.signal as sgl
from os import path
import scipy.optimize as op

### importing the omnitool package functions
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from omnitool import explore_plot
from mcmc import get_mcmc_sampler, mcmc_results, mcmc_sampler
from ptmcmc import get_ptmcmc_sampler, ptmcmc_results, ptmcmc_sampler, ptmcmc_plots, save_ptmcmc_sampler
from models import model_2exp
from psd import psd
from chi2 import chi2_freq, opt_chi2_freq

# close all plots
plt.close('all')

# creating time array
fs = 1e3
t_range = np.arange(0, 1, fs**-1)

# model function to study
funk = lambda t, p: model_2exp(*p, t0=t)(t_range)

# DATA
sig = 1e-6
pc = (1e-4, 1e-5, 1e-2, 1e-1, 5e-3)
data1 = funk(.5, pc) + np.random.normal(0, sig, t_range.shape)
data2 = funk(.4, pc) + np.random.normal(0, sig, t_range.shape)
data3 = funk(.6, pc) + np.random.normal(0, sig, t_range.shape)
data_list = (data1, data2, data3)


# plotting experimental data
for d in data_list:
    explore_plot(t_range, d, label='Exp. Pulse\n{}'.format(pc),
                 num='test explore_plot')

# NOISE LEVEL
noise_list = list()
for k in xrange(100):
    freq, noi = psd(np.fft.fft(np.random.normal(0, sig, t_range.shape)), fs)
    noise_list.append(noi)
npsd = np.mean(noise_list, axis=0)

# plotting noise level
plt.plot(freq, npsd, 'r--', label='Average Noise')
plt.legend()

# MODEL INIT
pinit =(2e-4, 0.1e-5, 10e-2, 3e-1, 1e-3)
mod = funk(.5, pinit)

# plotting experimental data
explore_plot(t_range, mod, label='Init. Mod. Pulse\n{}'.format(pinit),
             num='test explore_plot', alpha=0.2)

# chi2 freq init
fftdata = np.fft.fft(data1)
fftmod = np.fft.fft(mod)
x2 = chi2_freq(fftdata, fftmod, npsd, fs)

def aux_mini(p):
    p = np.power(10, p)
    fftmod = np.fft.fft(funk(0.5, p))
    x2 = chi2_freq(fftdata, fftmod, npsd, fs)
    return x2

#result = op.minimize(aux_mini, pinit, method='nelder-mead')
#popt = result.x
#mod_opt = funk(0.5, popt)
#explore_plot(t_range, mod_opt, label='Opt Mod. Pulse\n{}'.format(popt),
#             num='test explore_plot', ls='--', color='red', lw=3.0)

## opt chi2 freq init
#fftdata1 = np.fft.fft(data1)
#fftdata2 = np.fft.fft(data2)
#fftdata3 = np.fft.fft(data3)
#opt_mod = lambda t: funk(t, pinit)
#x2 = opt_chi2_freq(fftdata1, opt_mod, npsd, fs, (0, 1.), debug=True)
#
#def aux(p):
#    p = np.power(10, p)
#    opt_mod = lambda t: funk(t, p)
#    x2_tot= 0
#    topt_list = list()
#    for fd in (fftdata1, fftdata2, fftdata3):
#        x2, topt = opt_chi2_freq(fd, opt_mod, npsd, fs, (0., 1.))
#        x2_tot += x2
#        topt_list.append(topt)
#
#    print '\rChi2 = {:.2f}'.format(x2_tot),
#    return x2_tot, topt_list
#
#aux_mini = lambda p: aux(p)[0]

#result = op.minimize(aux_mini, np.log10(pinit), method='nelder-mead')
#popt = result.x
#
#x2_tot, to = aux(popt)
#popt = np.power(10, popt)
#
#for t in to:
#    mod_opt = funk(t, popt)
#    explore_plot(t_range, mod_opt, label='Opt Mod. Pulse\n{}'.format(popt),
#                 num='test explore_plot', ls='--', color='red', lw=3.0)

# XXX MCMC
# save directory
sampler_path = 'mcmc_sampler/autosave'

# running the mcmc analysis
bounds = ((-4.5, 3.5), (-5.5, -4.5), (-2.5, -1.5), (-1.5, -0.5), (-3, -2))
sampler = mcmc_sampler(aux_mini, bounds, nsteps=1000, path=sampler_path)

#    # loading the mcmc results
logd, chain, lnprob, acc = get_mcmc_sampler(sampler_path)

#    LAB = ('$log\ a$', '$log\ t$', '$log\ s$')
LAB = ('$log\ a1$', '$log\ a2$', '$log\ t1$', '$log\ t2$', '$log\ s$')
dim = int(logd['dim'])
xopt, inf, sup = mcmc_results(dim, chain, lnprob, acc, LAB)

popt, pinf, psup = map(lambda x: np.power(10,x),
                       (xopt, inf, sup))

## XXX PTMCMC
## save directory
#sampler_path = 'ptmcmc_sampler/autosave_2'
#
## running the mcmc analysis
##    bounds = ((0., 4.), (-6, 0.), (-6, 0.))
##    bounds = ((-4.5, 3.5), (-5.5, -4.5), (-2.5, -1.5), (-1.5, -0.5), (-3, -2))
#bounds = ((-5.5, -3.5), (-5.5, -3.5), (-2.5, -0.5), (-2.5, -0.5), (-3, -2))
##    bounds = ((-6, 0), (-6, 0), (-6, 0), (-6, 0), (-6, 0))
#condi = lambda p: p[2] < p[3]
#sampler = ptmcmc_sampler(aux_mini, bounds, nsteps=2000, ntemps=5, condi=condi)
##
##    # save sampler data
#save_ptmcmc_sampler(sampler, bounds, path = sampler_path)
#
## XXX analysing the ptmcmc results
##    # loading the mcmc results
#logd, chain, lnprob, acc = get_ptmcmc_sampler(sampler_path)
##
##    LAB = ('$log\ a$', '$log\ t$', '$log\ s$')
#LAB = ('$log\ a1$', '$log\ a2$', '$log\ t1$', '$log\ t2$', '$log\ s$')
#dim = int(logd['dim'])
#ntemps = int(logd['ntemps'])
#
#ptmcmc_plots(ntemps, dim, chain, lnprob, acc, LAB)
#
#xopt, inf, sup = ptmcmc_results(dim, chain, lnprob, acc, LAB)
#
#popt, pinf, psup = map(lambda x: np.power(10,x),
#                       (xopt, inf, sup))

### PLOT OPT
MOD = funk(0.5, popt)
MOD_lab = 'param= {}'.format(popt)

explore_plot(t_range, MOD,
             num='test explore_plot', label=MOD_lab)
