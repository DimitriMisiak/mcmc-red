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

nsample = 1000

ndim = 3

mu_generator = np.random.uniform(-1, 5, ndim)
mu = 10**mu_generator
#mu = np.random.uniform(int(1e-1), int(1e5), ndim)

sigma = mu/10
#sigma = np.random.uniform(0, 10, ndim)


print "Generating blob at mu={0} and sigma={1}".format(mu, sigma)

blob = np.random.normal(mu, sigma, (nsample, ndim))

print "Checking"
print "mean =", np.mean(blob, axis=0)
print "std =", np.std(blob, axis=0)

def chi2(param):
    return chi2_simple(blob, param, sigma)

# XXX MCMC
# save directory
sampler_path = 'mcmc_sampler/autosave'

# running the mcmc analysis
bounds = ((0, 1e6),) * ndim
sampler = mcmc_sampler(chi2, bounds, nsteps=10000, path=sampler_path)

#    # loading the mcmc results
logd, chain, lnprob, acc = get_mcmc_sampler(sampler_path)

lab = tuple(['$\mu${}'.format(i) for i in range(ndim)])

dim = int(logd['dim'])
xopt, inf, sup = mcmc_results(dim, chain, lnprob, acc, lab)

print xopt, inf, sup
