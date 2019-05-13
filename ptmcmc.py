#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Handy PTMCMC scripts.

Author:
    Dimitri Misiak (misiak@ipnl.in2p3.fr)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgl
import emcee
import corner
import os
import __main__

import sys
from os import path
sys.path.append( path.dirname( path.abspath(__file__) ) )
from savesys import savetxt, loadtxt


def chi2(time, data_1, data_2, noise=1):
    """Computes the chi2 function in frequency space.
    OLD FUNCTION NEEDED IN ANCIENT TEST FILES. DO NOT USE !!
    Parameters
    ----------
    data_1 : array_like
        First data array to be compared.
    data_2 : array_like
        Second data_array to be compared.
    noise : array_like or float
        PSD of the noise affecting the data.

    Returns
    -------
    x2 : float
        Value of the chi2 function. It should tends to dof if the data and
        noise are compatible.
    dof : int
        Degrees Of Freedom. Use to compute a normalized chi2.
    """
    # computes the psd correction coefficient
    t_step = time[1] - time[0]
    t_len = time[-1] - time[0]
    psd_corr = 2 * t_step**2 / t_len

    # computes the fft of the data
    # removing the frequency 0, and cropping the negative frequencies
    # WILL NOT WORK FOR t_len > 0.999 !!!
    ind = int(t_step**-1 /2) +1
    d1 = np.fft.fft(data_1)[1:ind]
    d2 = np.fft.fft(data_2)[1:ind]

    # noise psd
    J = np.array(noise)

    # computes the chi2
    x2 = np.sum( psd_corr * np.abs(d1-d2)**2 / J )

    # computes the dof
    dof = len(d1)

    return x2, dof

def chi2_new(time, data_1, data_2, noise=1):
    """Computes the chi2 function in frequency space.
    OLF FUNCTION NEEDED IN ANCIENT TEST FILES . DO NOT USE !!
    Parameters
    ----------
    data_1 : array_like
        First data array to be compared.
    data_2 : array_like
        Second data_array to be compared.
    noise : array_like or float
        PSD of the noise affecting the data.

    Returns
    -------
    x2 : float
        Value of the chi2 function. It should tends to dof if the data and
        noise are compatible.
    dof : int
        Degrees Of Freedom. Use to compute a normalized chi2.
    """
    # computes the psd correction coefficient
    t_step = time[1] - time[0]
    t_len = time[-1] - time[0]
    psd_corr = 2 * t_step**2 / t_len

    # noise psd
    J = np.array(noise)

    # computes the chi2
    x2 = np.sum( psd_corr * np.abs(data_1-data_2)**2 / J )

    # computes the dof
    dof = len(data_1)

    return x2, dof

def ptmcmc_sampler(aux, bounds, nsteps, ntemps, nwalkers=None,
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
    ntemps = ntemps

    pos_temp = list()
    for k in xrange(ntemps):

        pos = list()
        for n in xrange(nwalkers):
            accept = False
            while not accept:
                new_pos = [
                    np.random.uniform(low=l, high=h) for l,h in zip(binf, bsup)
                ]
                accept = condi(new_pos)
            pos.append(new_pos)

        pos_temp.append(pos)

#    pos_temp = np.random.uniform(low=-6.0, high=0.0, size=(ntemps, nwalkers, ndim))

    pos_temp = np.array(pos_temp)
#    print 'pos_temp.shape =', pos_temp.shape

    # MCMC analysis
#    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike)
    logp = lambda x: 0.0
    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, loglike, logp)
    sampler.run_mcmc(pos_temp, nsteps, rstate0=np.random.get_state())

    return sampler

def save_ptmcmc_sampler(sampler, bounds, path='mcmc_sampler/autosave'):
    """ Save the data contained in the ptmcmc sampler.
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

    # saving the markov chain
    with file(os.path.join(path,'chain.dat'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(sampler.chain.shape))
        for data_temp in sampler.chain:
            for data_slice in data_temp:
                np.savetxt(outfile, data_slice)
                outfile.write('# Next walker\n')
            outfile.write('# Next temperature\n')

    # saving the lnprob
    lnprob = sampler._lnprob
    with file(os.path.join(path,'lnprob.dat'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(lnprob.shape))
        for data_temp in lnprob:
            np.savetxt(outfile, data_temp)
            outfile.write('# Next temperature\n')

    # saving the acceptance fraction
    acc = sampler.acceptance_fraction
    with file(os.path.join(path,'acceptance.dat'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(acc.shape))
        for data_temp in acc:
            np.savetxt(outfile, data_temp)
            outfile.write('# Next temperature\n')

    entries = ('source', 'bounds',
               'dim', 'iterations', 'nwalkers', 'ntemps')

    try:
        source = __main__.__file__
    except:
        source = os.getcwd()

    values = (source, bounds,
              sampler.dim, sampler.chain.shape[2], sampler.nwalkers,
              sampler.ntemps)

    savetxt(entries, values, fpath=os.path.join(path ,'log.dat'))


def get_ptmcmc_sampler(sdir):
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
    ntemps = int(logd['ntemps'])

    # read chain
    chain = np.loadtxt(chainpath)
    chain = chain.reshape((ntemps, nwalkers, nsteps, ndim))
#    chain = chain.reshape((ntemps * nwalkers, nsteps, ndim))

    # read lnprob
    lnprob = np.loadtxt(lnpath)
    lnprob = lnprob.reshape((ntemps, nwalkers, nsteps))

    beta = emcee.ptsampler.default_beta_ladder(ndim, ntemps)
    beta = beta.reshape( (ntemps, 1, 1) )
    lnprob = lnprob / beta
#    lnprob = lnprob.reshape((ntemps * nwalkers, nsteps))

    # read acceptance
    acc = np.loadtxt(accpath)
    acc = acc.reshape((ntemps, nwalkers))

    return logd, chain, lnprob, acc


def ptmcmc_plots(ntemps, ndim, chain, lnprob, acc, labels):

    cmap = plt.get_cmap('jet')
    cmap = cmap(np.linspace(0., 1., ntemps))

    for k in reversed(xrange(ntemps)):

        ch = chain[k]
        ln = lnprob[k]
        ac = acc[k]
        c = cmap[k]

        plt.figure('Acceptance fraction')
        plt.plot(ac, color=c)

        # CONVERGENCE plot
        fig = plt.figure(num='CONVERGENCE', figsize=(7,8))
        ax = fig.get_axes()
        # Check if the figure was already plotted.
        if not len(ax):
            fig, ax = plt.subplots(ndim+1, 1, sharex=True, figsize=(7, 8),
                                   num='CONVERGENCE')

        ax[-1].set_xlabel('Iterations')
        ax[-1].set_yscale('log')
        for a, l in zip(ax, labels + ('lnprob',)):
            a.set_ylabel(l)
            a.grid()

        # plotting the lnprob array and the cut threshold
        ax[-1].plot(-ln.T, color=c)

        # loop over the chains
        for chk, lnk in zip(ch, ln):
            for n in range(ndim):
                # plotting the accepted chain and their respective burnin
                ax[n].plot(chk[:,n].T, color=c, lw=0.1, alpha=1.0)
                ax[n].scatter([0], chk[0, n], color=c, marker='>')

        fig.tight_layout(h_pad=0.0)

        samples = reduce(lambda a,b: np.append(a,b, axis=0), ch)

        best_ind = np.unravel_index(ln.argmax(), ln.shape)
        best_chi2 = -2 * ln[best_ind]
        xopt = ch[best_ind]

        if k == ntemps-1:
            fig_corner = corner.corner(samples, bins=50, smooth=1, color=c,
                                labels=['{}'.format(l) for l in labels],
                                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                truths=xopt,
                                title_kwargs={"fontsize": 12})
        else:
            corner.corner(samples, bins=50, smooth=1, color=c,
                                labels=['{}'.format(l) for l in labels],
                                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                truths=xopt,
                                title_kwargs={"fontsize": 12},
                                fig = fig_corner)


def ptmcmc_results(ndim, chain, lnprob, acc, labels):

    # chains of lowest temperature
    chain = chain[0]
    lnprob = lnprob[0]
    acc = acc[0]

    # acceptance fraction cut
    tracc = (0.15, 0.8)
    ind = np.where(np.logical_or(acc < tracc[0], acc > tracc[1]))
    bam = chain[ind]
    chain = np.delete(chain, ind, axis=0)
    lnprob = np.delete(lnprob, ind, axis=0)

#    print 'shape chain:', chain.shape
#    print 'shape lnprob:', lnprob.shape

    plt.figure('Acceptance fraction Results')
    plt.plot(acc)
    for thresh in tracc:
        plt.axhline(thresh, color='r', ls='--')

    # CONVERGENCE plot
    fig, ax = plt.subplots(ndim+1, 1, sharex=True, figsize=(7, 8),
                           num='CONVERGENCE RESULTS')
    ax[-1].set_xlabel('Iterations')
    ax[-1].set_yscale('log')
    for a, l in zip(ax, labels + ('lnprob',)):
        a.set_ylabel(l)
        a.grid()

    # loop over the parameters
    for n in range(ndim):

        if len(bam) > 0:
            # plotting the chains discarded by the acceptance cut
            ax[n].plot(bam[:, :, n].T, color='r', lw=1., alpha=0.4)

    # by default : no cut

    burnin_list = np.ones(lnprob.shape[0], dtype='int') * 0

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
#    lncut = -300
    ax[-1].axhline(-lncut, color='r')

    burnin_list = list()
    safe_burn = 100
    for lnk in lnprob:
        try:
            burn = np.where(lnk <  lncut)[0][-1] + safe_burn
        except:
            print 'Could not apply convergence cut properly'
            burn = safe_burn
        burnin_list.append(burn)

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
            ax[n].plot(ite_ok, chk_ok[:,n].T, color='b', lw=0.1, alpha=1.0)
            ax[n].plot(ite_no, chk_no[:,n].T, color='k', lw=0.1, alpha=0.4)
            ax[n].scatter([0], chk[0, n], color='r', marker='>')

        # plotting converged chain lnprob
        ax[-1].plot(ite_ok, -lnk_ok.T, color='b', lw=0.1, zorder=10)

    fig.tight_layout(h_pad=0.0)

    samples = reduce(lambda a,b: np.append(a,b, axis=0), chain_ok_list)

    best_ind = np.unravel_index(lnprob.argmax(), lnprob.shape)
    best_chi2 = -2 * lnprob[best_ind]
    xopt = chain[best_ind]


#    # checking the correlation
#    fig, ax = plt.subplots(2, sharex=True)
#    for a in ax:
#        a.grid()
#        a.set_xscale('log')
#    ax[1].set_xlabel('Iterations')
#    ax[1].set_ylabel('Corr p0')
#    ax[0].set_ylabel('p0')
#
#    for c in chain[:,:,0]:
#        funk = emcee.autocorr.function(c)
#        ax[0].plot(c)
#        ax[1].plot(funk)

    # CORNER plot
#    aux_fig, ax = plt.subplots(ndim,ndim,num='CORNER', figsize=(ndim,ndim))
    fig = corner.corner(samples, bins=50, smooth=1,
                        labels=['{}'.format(l) for l in labels],
                        quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        truths=xopt,
                        title_kwargs={"fontsize": 12})#, fig=aux_fig)
    fig.tight_layout()

    # quantiles of the 1d-histograms
    inf, med, sup = np.percentile(samples, [16, 50, 84], axis=0)

    # Analysis end message
    print "PTMCMC results :"
    for n in range(ndim):
        print labels[n]+'= {:.2e} + {:.2e} - {:.2e}'.format(
            med[n], sup[n]-med[n], med[n]-inf[n]
        )
    for n in range(ndim):
        print labels[n]+'\in [{:.3e} , {:.3e}] with best at {:.3e}'.format(
                inf[n], sup[n], xopt[n]
        )
    if not np.all(np.logical_and(inf<xopt, xopt<sup)):
        print 'Good luck fixing that :P'

    print 'Chi2 = {}'.format(best_chi2)

    return xopt, inf, sup
