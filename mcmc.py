#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Handy MCMC scripts.

Transition to full frequency strategy.

Author:
    Dimitri Misiak (misiak@ipnl.in2p3.fr)
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import os
import __main__

import sys
from os import path
sys.path.append( path.dirname( path.abspath(__file__) ) )
from savesys import savetxt, loadtxt


def mcmc_sampler(aux, bounds, nsteps, nwalkers=None,
                 path='mcmc_sampler/autosave', save=True,
                 condi=None):
    """ MCMC Analysis routine. Log scale seach in parameter space.

    Parameters
    ----------
    aux : function
        Minimized function. Should be a frequency chi2 function
        for proper results.
    bounds: array_like of tuple of 2 floats
        Starting parameter set for MCMC analysis.
    nsteps : int
        Number of steps.
    nwalkers : None or int, optional
        Numbers of walkers. Should not be inferior to 2 times
        the number of parameters. By default, set to 10 times
        the number of parameters.
    savename : str, optional
        Path the save directory
    Returns
    -------
    sampler : emcee.ensemble.EnsembleSampler
        Object manipulated by the mcmc. Has several class attributes which
        contain the Markov chain, the lnprob list, and other characteristics
        of the mcmc analysis.
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

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
            return -0.5*aux(x)
        else:
            return -np.inf

    # number of parameters/dimensions
    ndim = len(bounds)

    # default nwalkers
    if nwalkers == None:
        nwalkers = 10 * ndim

    # walkers are uniformly spread in the parameter space
    pos = list()
    for n in range(nwalkers):
        accept = False
        while not accept:
            new_pos = [
                np.random.uniform(low=l, high=h) for l,h in zip(binf, bsup)
            ]
            accept = condi(new_pos)
        pos.append(new_pos)

    # MCMC analysis
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike)
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())


    # saving the markov chain
    with open(os.path.join(path,'chain.dat'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(sampler.chain.shape))
        for data_slice in sampler.chain:
            np.savetxt(outfile, data_slice)
            outfile.write('# Next walker\n')

    # saving the lnprob
    lnprob = sampler._lnprob
    with open(os.path.join(path,'lnprob.dat'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(lnprob.shape))
        np.savetxt(outfile, lnprob)

    # saving the acceptance fraction
    acc = sampler.acceptance_fraction
    with open(os.path.join(path,'acceptance.dat'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(acc.shape))
        np.savetxt(outfile, acc)

    entries = ('source', 'bounds',
               'dim', 'iterations', 'nwalkers')

    try:
        source = __main__.__file__
    except:
        source = os.getcwd()

    values = (source, bounds,
              sampler.dim, sampler.iterations, sampler.k)

    savetxt(entries, values, fpath=os.path.join(path ,'log.dat'))

    return sampler


def mcmc_sampler_multi(lnpostfn, bounds, nsteps, nwalkers=None,
                 path='mcmc_sampler/autosave', save=True,
                 condi=None, threads=1, scale='linear'):
    """ MCMC Analysis routine. Log scale seach in parameter space.

    Parameters
    ----------
    lnpostfn : function
        Loglikelihood of Minimized function. See emcee module for more info.
    bounds: array_like of tuple of 2 floats
        Starting parameter set for MCMC analysis.
    nsteps : int
        Number of steps.
    nwalkers : None or int, optional
        Numbers of walkers. Should not be inferior to 2 times
        the number of parameters. By default, set to 10 times
        the number of parameters.
    savename : str, optional
        Path the save directory
    threads : int
        Number of threads for the multiprocessing.
    scale : str, optional
        Scale for the spreading of the initial markov chains.
        Can be either 'linear' or 'log'.

    Returns
    -------
    sampler : emcee.ensemble.EnsembleSampler
        Object manipulated by the mcmc. Has several class attributes which
        contain the Markov chain, the lnprob list, and other characteristics
        of the mcmc analysis.
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

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

    # number of parameters/dimensions
    ndim = len(bounds)

    # default nwalkers
    if nwalkers == None:
        nwalkers = 10 * ndim

    # walkers are uniformly spread in the parameter space
    # according to the search scale
    pos = list()
    for n in range(nwalkers):
        accept = False
        while not accept:
            if scale == 'linear':
                new_pos = [
                    np.random.uniform(low=l, high=h) for l,h in zip(binf, bsup)
                ]
                accept = condi(new_pos)
            elif scale == 'log':
                new_pos = [
                    10**np.random.uniform(low=l, high=h) for l,h in zip(np.log10(binf), np.log10(bsup))
                ]
                accept = condi(new_pos)
            else:
                raise Exception('Scale parameter not set correctly. Should be \'linear\' or \'log\'')
        pos.append(new_pos)

    # MCMC analysis
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpostfn, threads=threads)
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())


    # saving the markov chain
    with open(os.path.join(path,'chain.dat'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(sampler.chain.shape))
        for data_slice in sampler.chain:
            np.savetxt(outfile, data_slice)
            outfile.write('# Next walker\n')

    # saving the lnprob
    lnprob = sampler._lnprob
    with open(os.path.join(path,'lnprob.dat'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(lnprob.shape))
        np.savetxt(outfile, lnprob)

    # saving the acceptance fraction
    acc = sampler.acceptance_fraction
    with open(os.path.join(path,'acceptance.dat'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(acc.shape))
        np.savetxt(outfile, acc)

    entries = ('source', 'bounds',
               'dim', 'iterations', 'nwalkers')

    try:
        source = __main__.__file__
    except:
        source = os.getcwd()

    values = (source, bounds,
              sampler.dim, sampler.iterations, sampler.k)

    savetxt(entries, values, fpath=os.path.join(path ,'log.dat'))

    return sampler


def get_mcmc_sampler(sdir):
    """ Read the sampler info from disk created by mcmc_sampler.

    Parameters
    ----------
    sdir : str
        Save directory path.

    Returns
    -------
    logd : dict
        Contains the characteristics of the mcmc analysis.
    chain : ndarray
        Array of shape (nwalkers, nsteps, ndim). Contains the positions of
        all walkers for each iterations.
    lnprob : ndarray
        Array of shape (nwalkers, nsteps). Contains the log probability
        of all walkers for each iterations.
    """
    # log chain, lnprob file path
    logpath = os.path.join(sdir, 'log.dat')
    chainpath = os.path.join(sdir, 'chain.dat')
    lnpath = os.path.join(sdir, 'lnprob.dat')
    accpath = os.path.join(sdir, 'acceptance.dat')

    # read log and extracts info into dict
    entries, values = loadtxt(logpath)
    logd = dict([(e,v) for e,v in zip(entries, values)])

    # extracting shape of the chain and lnprob arrays
    ndim = int(logd['dim'])
    nwalkers = int(logd['nwalkers'])
    nsteps = int(logd['iterations'])

    # read chain
    chain = np.loadtxt(chainpath)
    chain = chain.reshape((nwalkers, nsteps, ndim))

    # read lnprob
    lnprob = np.loadtxt(lnpath)
    lnprob = lnprob.reshape((nwalkers, nsteps))

    # read acceptance
    acc = np.loadtxt(accpath)

    return logd, chain, lnprob, acc


def mcmc_results(ndim, chain, lnprob, acc, labels, scale='linear', savedir=None):
    """ Plot the results of the mcmc analysis, and return these results.

    Parameters
    ----------
    .
    .
    .
    scale : str, optional
        Scale for the spreading of the initial markov chains.
        Can be either 'linear' or 'log'.
    savedir : str, optional
        If given, save the figure to the savedir. By default, set to None,
        figures are not saved.

    Returns
    -------
    xopt : numpy.ndarray
        Optimal parameters, minimising the loglikelihood.
    inf : numpy.ndarray
        Lower 1-sigma bounds for the optimal parameters.
    sup : numpy.ndarray
        Upper 1-sigma bounds for the optimal parameters.
    """
    # acceptance fraction cut
    tracc = (0.2, 0.8)
    ind = np.where(np.logical_or(acc < tracc[0], acc > tracc[1]))
    bam = chain[ind]
    chain = np.delete(chain, ind, axis=0)
    lnprob = np.delete(lnprob, ind, axis=0)

    print('shape chain:'), chain.shape
    print('shape lnprob:'), lnprob.shape

    fig_acceptance = plt.figure('ACCEPTANCE FRACTION')
    plt.bar(np.arange(acc.shape[0]), acc)
    for thresh in tracc:
        plt.axhline(thresh, color='r', ls='--')
    plt.xlabel('Marker Chain Index')
    plt.ylabel('Acceptance fraction')
    plt.ylim(0., 1.)
    plt.tight_layout()

    ### CONVERGENCE plot
    fig_convergence, ax = plt.subplots(ndim+1, 1, sharex=True, figsize=(7, 8),
                           num='CONVERGENCE')
    ax[-1].set_xlabel('Iterations')
    ax[-1].set_yscale('log')
    ax[-1].set_xscale('log')
    for a, l in zip(ax, labels + ('lnprob',)):

        a.set_ylabel(l)
        a.grid()

    # loop over the parameters
    for n in range(ndim):
        if scale == 'log' :
            ax[n].set_yscale('log')
        if len(bam) > 0:
            # plotting the chains discarded by the acceptance cut
            ax[n].plot(bam[:, :, n].T, color='r', lw=1., alpha=0.4)

#    # convergence cut with mean
#    lnlncut = np.mean(np.log10(-lnprob))
#    burnin_list = list()
#    for lnk in lnprob:
#        try:
#            burn = np.where(np.log10(-lnk) > lnlncut)[0][-1] + 100
#        except:
#            burn = 0
#        burnin_list.append(burn)
#
#    ax[-1].axhline(np.power(10,lnlncut), color='r')

    # convergence cut with best prob
    lncut = 1.1 * lnprob.max()
    burnin_list = list()
    for lnk in lnprob:
        try:
            burn = np.where(lnk <  lncut)[0][-1] + 100
        except:
            print('Could not apply convergence cut properly')
            burn = 0
        burnin_list.append(burn)

    ax[-1].axhline(-lncut, color='r')

    # plotting the log10(-lnprob) array and the cut threshold
    ax[-1].plot(-lnprob.T, color='k')

    chain_ok_list = list()
    # loop over the chains
    for chk, brn, lnk in zip(chain, burnin_list, lnprob):

        # iterations array
        ite = range(chk.shape[0])

        # converged chain and saving it
        ite_ok = ite[brn:]
        chk_ok = chk[brn:, :]
        lnk_ok = lnk[brn:]
        chain_ok_list.append(chk_ok)

        # not converged chain
        ite_no = ite[:brn]
        chk_no = chk[:brn, :]

        # loop over the parameters
        for n in range(ndim):

            # plotting the accepted chain and their respective burnin
            ax[n].plot(ite_ok, chk_ok[:,n].T, color='b', lw=1., alpha=1.)
            ax[n].plot(ite_no, chk_no[:,n].T, color='k', lw=1., alpha=0.4)
            ax[n].scatter([0], chk[0, n], color='r', marker='o')

        # plotting converged chain lnprob
        ax[-1].plot(ite_ok, -lnk_ok.T, color='b')

    fig_convergence.tight_layout(h_pad=0.0)

    # samples = reduce(lambda a,b: np.append(a,b, axis=0), chain_ok_list)
    samples = np.vstack(chain_ok_list)
    
    best_ind = np.unravel_index(lnprob.argmax(), lnprob.shape)
    best_chi2 = -2 * lnprob[best_ind]
    xopt = chain[best_ind]


    ### CORRELATION plot
    fig_correlation, ax = plt.subplots(2, sharex=True, num='CORRELATION')
    for a in ax:
        a.grid()
        a.set_xscale('log')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Corr p0')
    ax[0].set_ylabel('p0')

    for c in chain[:,:,0]:
        funk = emcee.autocorr.function(c)
        ax[0].plot(c)
        ax[1].plot(funk)


    ### CORNER plot
#    aux_fig, ax = plt.subplots(ndim,ndim,num='CORNER', figsize=(ndim,ndim))
    if scale == 'linear':
        fig_corner = corner.corner(
                samples,
                bins=50, smooth=1,
                labels=labels,
                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                truths=xopt,
                title_kwargs={"fontsize": 12}
        )

    elif scale == 'log':
        fig_corner = corner.corner(
                np.log10(samples),
                bins=50, smooth=1,
                labels=['log({})'.format(l) for l in labels],
                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                truths=np.log10(xopt),
                title_kwargs={"fontsize": 12}
        )

    fig_corner.tight_layout()

    # quantiles of the 1d-histograms
    inf, med, sup = np.percentile(samples, [16, 50, 84], axis=0)

    # Analysis end message
    print("MCMC results :")
    for n in range(ndim):
        print(labels[n]+'= {:.2e} + {:.2e} - {:.2e}'.format(
            med[n], sup[n]-med[n], med[n]-inf[n]
        ))
    for n in range(ndim):
        print(labels[n]+'\in [{:.3e} , {:.3e}] with best at {:.3e}'.format(
                inf[n], sup[n], xopt[n]
        ))
    if not np.all(np.logical_and(inf<xopt, xopt<sup)):
        print('Optimal parameters out the 1-sigma range ! Good luck fixing that :P')

    print('Chi2 = {}'.format(best_chi2))

    if savedir is not None:
        fig_acceptance.savefig(savedir+'/acceptance.png')
        fig_convergence.savefig(savedir+'/convergence.png')
        fig_correlation.savefig(savedir+'/correlation.png')
        fig_corner.savefig(savedir+'/corner.png')

    return xopt, inf, sup
