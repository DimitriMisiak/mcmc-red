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

from mcmc import get_mcmc_sampler, mcmc_results, mcmc_sampler
from models import model_2exp
from psd import psd
from chi2 import chi2_freq, opt_chi2_freq, chi2_simple

# close all plots
plt.close('all')

nsample = 100
#mu1, sigma1 = 1., 1.
mu1 = np.random.uniform(-10, 10)
sigma1 = np.random.uniform(0, 10)
#mu2, sigma2 = 2., 5.
mu2 = np.random.uniform(-10, 10)
sigma2 = np.random.uniform(0, 10)

print ("Generating blob at (mu1, mu2)=({0:.2f}, {1:.2f})"
       " and (sigma1, sigma2)=({2:.2f}, {3:.2f})".format(mu1, mu2, sigma1, sigma2))

blob = np.random.normal((mu1, mu2), (sigma1, sigma2), (nsample,2))

print("Checking")
print("mean =", np.mean(blob, axis=0))
print("std =", np.std(blob, axis=0))

def chi2(param):
    return chi2_simple(blob, param, (sigma1, sigma2))

# XXX MCMC
# save directory
sampler_path = 'mcmc_sampler/autosave'

# running the mcmc analysis
bounds = ((-20, 20),(-20, 20))
sampler = mcmc_sampler(chi2, bounds, nsteps=1000, path=sampler_path)

#    # loading the mcmc results
logd, chain, lnprob, acc = get_mcmc_sampler(sampler_path)

#    LAB = ('$log\ a$', '$log\ t$', '$log\ s$')
#LAB = ('$log\ a1$', '$log\ a2$', '$log\ t1$', '$log\ t2$', '$log\ s$')
lab = ('$\mu1$','$\mu2$')
dim = int(logd['dim'])
xopt, inf, sup = mcmc_results(dim, chain, lnprob, acc, lab)

print(xopt, inf, sup)
