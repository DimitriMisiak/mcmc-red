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
from mcmc import get_mcmc_sampler, mcmc_results, mcmc_sampler, mcmc_sampler_multi
from models import model_2exp
from psd import psd
from chi2 import chi2_freq, opt_chi2_freq, chi2_simple
import emcee

# close all plots
plt.close('all')

nsample = 1000

ndim = 4

SCALE = 'log'

### LINEAR SCALE
if SCALE == 'linear':
    mu = np.random.uniform(-10, 10, ndim)
    sigma = np.random.uniform(0, 10, ndim)
    bounds = ((-20, 20),) * ndim

### LOG SCALE
elif SCALE == 'log':
    mu_generator = np.random.uniform(-6, 0, ndim)
    mu = 10**mu_generator
    sigma = mu/10
    bounds = ((1e-7, 1e1),) * ndim

else:
    raise Exception('SCALE not set properly!')

print "Generating blob at mu={0} and sigma={1}".format(mu, sigma)

blob = np.random.normal(mu, sigma, (nsample, ndim))

print "Checking"
print "mean =", np.mean(blob, axis=0)
print "std =", np.std(blob, axis=0)

def chi2(param):
    return chi2_simple(blob, param, sigma)

#def chi2(param):
#    x2 = np.sum( (blob - np.array(param))**2  / np.array(sigma)**2 )
#    return x2

condi = None
# XXX MCMC
# save directory
sampler_path = 'mcmc_sampler/autosave'

# extracts the sup bounds and the inf bounds
bounds = list(bounds)
binf = list()
bsup = list()
for b in bounds:
    inf, sup = b
    binf.append(inf)
    bsup.append(sup)
binf = np.array(binf)
bsup = np.array(bsup)

# additionnal constrain as function of the parameters
if condi == None:
    condi = lambda p: True

# Loglikelihood function taking into accounts the bounds
def loglike(x):
    """ Loglikelihood being -chi2/2.
    Take into account the bounds.
    """
    cinf = np.sum(x<binf)
    csup = np.sum(x>bsup)
    if cinf == 0 and csup == 0 and condi(x) == True:
#            return -0.5*aux(np.power(10,x))
        return -0.5*chi2(x)
    else:
        return -np.inf

# running the mcmc analysis
sampler = mcmc_sampler_multi(loglike, bounds, nsteps=1000, path=sampler_path, threads=2, scale=SCALE)
#nwalkers=None
#nsteps=10000
#threads=4
##############################################################################
## extracts the sup bounds and the inf bounds
#bounds = list(bounds)
#binf = list()
#bsup = list()
#for b in bounds:
#    inf, sup = b
#    binf.append(inf)
#    bsup.append(sup)
#binf = np.array(binf)
#bsup = np.array(bsup)
#
#condi = None
## additionnal constrain as function of the parameters
#if condi == None:
#    condi = lambda p: True
#
## Loglikelihood function taking into accounts the bounds
#def loglike(x):
#    """ Loglikelihood being -chi2/2.
#    Take into account the bounds.
#    """
#    cinf = np.sum(x<binf)
#    csup = np.sum(x>bsup)
#    if cinf == 0 and csup == 0 and condi(x) == True:
##            return -0.5*aux(np.power(10,x))
#        return -0.5*chi2(x)
#    else:china moon
#        return -np.inf
#
## number of parameters/dimensions
#ndim = len(bounds)
#
## default nwalkers
#if nwalkers == None:
#    nwalkers = 10 * ndim
#
## walkers are uniformly spread in the parameter space
#pos = list()
#for n in xrange(nwalkers):
#    accept = False
#    while not accept:
#        new_pos = [
#            np.random.uniform(low=l, high=h) for l,h in zip(binf, bsup)
#        ]
#        accept = condi(new_pos)
#    pos.append(new_pos)
#
## MCMC analysis
#sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, threads=threads)
#sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())

#############################################################################
#    # loading the mcmc results
logd, chain, lnprob, acc = get_mcmc_sampler(sampler_path)

lab = tuple(['$\mu${}'.format(i) for i in range(ndim)])

dim = int(logd['dim'])
xopt, inf, sup = mcmc_results(dim, chain, lnprob, acc, lab,
                              scale=SCALE, savedir=sampler_path)

print xopt, inf, sup
