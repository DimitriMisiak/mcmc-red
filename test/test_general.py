#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:24:14 2019

@author: misiak
"""

import numpy as np

from mcmc_red.general import explore_plot


fs = int(1e3)
wlen = 1.

time_array = np.arange(0, wlen, fs**-1)

pulse_array = np.exp(-time_array/5e-2)

noise_array = np.random.normal(0, 0.1, size=time_array.shape)

explore_plot(
        time_array,
        pulse_array,
        label='pulse'
)

explore_plot(
        time_array,
        noise_array,
        label='noise'
)

print('Test is done!')
