#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:16:29 2018

Script gathering functions related to the psd and fft calculations.

@author: misiak
"""

import numpy as np

def psd(fft, fs, weight=None):
    """
    Computes the Power Spectral Density (PSD) from the Fast Fourier Transform
    (FFT given by numpy.fft.fft).

    Parameters
    ==========
    fft : nd.array
        FFT array whose frequency are ordered as the numpy.fft.fft function
        result (i.e. [0, positive, negative]).
    fs : float
        Sampling frequency.
    weight : None or array_like
        Weights of the frequencies in the psd calculation. If None, the
        weight are all 1 which correponds to the boxcar window.

    Returns
    =======
    freq : nd.array
        Frequency array containing only the positive frequencies (and the 0th).
    psd : nd.array
        PSD array.
    """
    nfft = fft.shape[0]
    if weight == None:
        s1 = nfft
        s2 = nfft
    else :
        s1 = np.sum(weight)
        s2 = np.sum(np.array(weight)**2)

    # Nyquist frequency
    #fny = float(fs) / 2

    # Frequency resolution
    #fres = float(fs) / nfft

    # Equivalent Noise BandWidth
    enbw = float(fs) * s2 / s1**2

    freq = np.fft.fftfreq(nfft, fs**-1)

    if nfft % 2:
        num_freqs = (nfft + 1)//2
    else:
        num_freqs = nfft//2 + 1
        # Correcting the sign of the last point
        freq[num_freqs-1] *= -1

    freq = freq[1:num_freqs]
    fft = fft[..., 1:num_freqs]

    psd_array = np.abs(fft)**2 / (enbw * s1**2)

    if nfft % 2:
        psd_array[..., :] *= 2
    else:
        # Last point is unpaired Nyquist freq point, don't double
        psd_array[..., :-1] *= 2

    return freq, psd_array


def angle_psd(fft):
    """
    Complex phase of the fft term, on the positive frequencies only.
    
    See also: inv_psd
    """
    nfft = fft.shape[0]
    if nfft % 2:
        num_freqs = (nfft + 1)//2
    else:
        num_freqs = nfft//2 + 1
    return np.angle(fft[..., 1:num_freqs])


def inv_psd(psd, fs, angle=None, mean=0):
    """ 
    Create a temporal array from the psd and the complex phase of the fft.
    If angle is set to None (by default), the phase are randomized, which
    gives a noise array.    
    """
    psd_array = np.array(psd)
    nfft = len(psd_array)*2
    #if weight == None:
    s1 = nfft
    s2 = nfft    
    # Equivalent Noise BandWidth
    enbw = float(fs) * s2 / s1**2

    # normalization
    psd_array[:-1] /= 2
    fft_all_freq  = np.sqrt( enbw * s1**2 * psd_array) 
    
    # phases
    phi_array = np.random.uniform(0, 2*np.pi, nfft//2-1)

    if angle is None:
        phi_array = np.random.uniform(0, 2*np.pi, nfft//2-1)
    else:
        assert len(psd) == len(angle)
        phi_array = angle[:-1]
    
    # mean at zero
    fft_0 = np.array([mean*nfft,])
    
    # negative frequency with last frequency
    fft_neg = np.array(fft_all_freq, dtype='complex')
    
    fft_neg[:-1] *= np.exp(-1j*phi_array)
    # positive frequency without last frequency
    fft_pos = np.conjugate(fft_neg)[:-1]
    
    # concatenating into fft_array
    fft_neg = fft_neg[::-1]
    fft_array = np.concatenate((fft_0, fft_pos, fft_neg))
    
    return np.fft.ifft(fft_array).real


def psd_from_fft2(fft2, fs, weight=None):
    
    nfft = fft2.shape[0]
    if weight == None:
        s1 = nfft
        s2 = nfft
    else :
        s1 = np.sum(weight)
        s2 = np.sum(np.array(weight)**2)

    # Nyquist frequency
    #fny = float(fs) / 2

    # Frequency resolution
    #fres = float(fs) / nfft

    # Equivalent Noise BandWidth
    enbw = float(fs) * s2 / s1**2

    freq = np.fft.fftfreq(nfft, fs**-1)

    if nfft % 2:
        num_freqs = (nfft + 1)//2
    else:
        num_freqs = nfft//2 + 1
        # Correcting the sign of the last point
        freq[num_freqs-1] *= -1

    freq = freq[1:num_freqs]
    fft2 = fft2[..., 1:num_freqs]

    psd_array = fft2 / (enbw * s1**2)

    if nfft % 2:
        psd_array[..., :] *= 2
    else:
        # Last point is unpaired Nyquist freq point, don't double
        psd_array[..., :-1] *= 2

    return freq, psd_array
 