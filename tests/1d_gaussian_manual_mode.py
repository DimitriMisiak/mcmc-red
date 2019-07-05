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

import mcmc_red as mcr

from tqdm import tqdm

import emcee

# close all plots
plt.close('all')

nsample = 100
mu, sigma = 1., 1.
#mu = np.random.uniform(-10, 10)
#sigma = np.random.uniform(0, 10)

print("Generating blob at mu={0:.2f} and sigma={1:.2f}".format(mu, sigma))
blob = np.random.normal(mu, sigma, nsample)

nsteps=10000
#progress_bar = tqdm(total=nsteps*10)

def chi2(param):
#    progress_bar.update()
    return mcr.chi2_simple(blob, param, sigma)
#
##### defining the MCMC Routine
#    
## save directory
#sampler_path = 'mcmc_sampler/autosave'
#
## running the mcmc analysis
#bounds = ((-20, 20),)
##sampler = mcr.mcmc_sampler(chi2, bounds, nsteps=nsteps, path=sampler_path)
#
#aux = chi2
#pos = None
#nwalkers = None
#
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
## additionnal constrain as function of the parameters
#condi = lambda p: True
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
#        return -0.5*aux(x)
#    else:
#        return -np.inf
#
## number of parameters/dimensions
#ndim = len(bounds)
#
## default nwalkers
#if nwalkers == None:
#    nwalkers = 10 * ndim
#
#if pos is not None:
#    assert len(pos) == nwalkers
#else:
#    # walkers are uniformly spread in the parameter space
#    pos = list()
#    for n in range(nwalkers):
#        accept = False
#        while not accept:
#            new_pos = [
#                np.random.uniform(low=l, high=h) for l,h in zip(binf, bsup)
#            ]
#            accept = condi(new_pos)
#        pos.append(new_pos)
#
## MCMC analysis
#sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike)
#
#sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state(), progress=True)



# XXX MCMC
# save directory
sampler_path = 'mcmc_sampler/autosave'

# running the mcmc analysis
bounds = ((-20, 20),)
sampler = mcr.mcmc_sampler(chi2, bounds, nsteps=nsteps, path=sampler_path,
                           progress=True)

#    # loading the mcmc results
logd, chain, lnprob, acc = mcr.get_mcmc_sampler(sampler_path)

#    LAB = ('$log\ a$', '$log\ t$', '$log\ s$')
#LAB = ('$log\ a1$', '$log\ a2$', '$log\ t1$', '$log\ t2$', '$log\ s$')
lab = ('$\mu$',)
dim = int(logd['dim'])
xopt, inf, sup = mcr.mcmc_results(dim, chain, lnprob, acc, lab)

print(xopt, inf, sup)
