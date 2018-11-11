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
from chi2 import chi2_freq, opt_chi2_freq, chi2_simple

# close all plots
plt.close('all')

nsample = 100
mu, sigma = 1., 1.
#mu = np.random.uniform(-10, 10)
#sigma = np.random.uniform(0, 10)

print "Generating blob at mu={0:.2f} and sigma={1:.2f}".format(mu, sigma)
blob = np.random.normal(mu, sigma, nsample)

def chi2(param):
    return chi2_simple(blob, param, sigma)

# XXX MCMC
# save directory
sampler_path = 'mcmc_sampler/autosave'

# running the mcmc analysis
bounds = ((-20, 20),)
sampler = mcmc_sampler(chi2, bounds, nsteps=1000, path=sampler_path)

#    # loading the mcmc results
logd, chain, lnprob, acc = get_mcmc_sampler(sampler_path)

#    LAB = ('$log\ a$', '$log\ t$', '$log\ s$')
#LAB = ('$log\ a1$', '$log\ a2$', '$log\ t1$', '$log\ t2$', '$log\ s$')
lab = ('$\mu$',)
dim = int(logd['dim'])
xopt, inf, sup = mcmc_results(dim, chain, lnprob, acc, lab)

print xopt, inf, sup
