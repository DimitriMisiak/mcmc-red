#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:16:01 2018

Template for a script meant to fit some data.

@author: misiak
"""

# IMPORT

# MAIN FUNCTIONS
def get_data():
    """ Reads the data files.
    """
    pass


def model():
    """ Model function used to fit the data.
    """
    pass


def comparator():
    """ Compares the data and the model for the given parameters.
    Return a value which quantifies the difference between model and data.
    """
    pass


def fit():
    """ Fit the data with the model.
    """
    pass


# LAUNCHER

if __name__ == '__main__':

    # get the data
    data = get_data()

    # plot the data if possible

    # define model
    mod = model()

    # plot the model for an initial set of paramaters

    # define the comparator and check the comparator on the data and initial model
    x2 = comparator()

    # fit the data with the model using the comparator
    xopt = fit()

    # plot the result, the analysis, etc...