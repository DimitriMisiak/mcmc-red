#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Handy scripting toolbox full of convenient functions.

Author:
    Dimitri Misiak (misiak@ipnl.in2p3.fr)
"""


import os
import re
import matplotlib.pyplot as plt
import scipy.signal as sgl

try:
    import Tkinter as Tk
    import tkFileDialog
    
    def choose_dir(initialdir=os.path.dirname(__file__)):
        """Returns the directory path selected within an explorer.
    
        Parameters
        ----------
        initialdir : str
            Initial path of the explorer. Default value set to the
            directory of this script.
    
        Returns
        -------
        dirpath : str
            Path of the selected directory.
        """
        root = Tk.Tk()
        dirpath = tkFileDialog.askdirectory(
            parent=root,
            initialdir=initialdir
        )
        root.destroy()
        return dirpath
    
    
    def choose_files(initialdir=os.path.dirname(__file__)):
        """Returns the path(s) of the files selected within an explorer.
    
        Parameters
        ----------
        initialdir : str
            Initial path of the explorer. Default value set to the
            directory of this script.
    
        Returns
        -------
        fpath : tuple
            Tuple containing the absolute path(s) of the selected files.
        """
        root = Tk.Tk()
        fpath = tkFileDialog.askopenfilenames(
            parent=root,
            initialdir=initialdir
        )
        root.destroy()
        return fpath
    
except:
    'Cannot import Tkinter'
    





def read_log(logpath):
    """
    Extracts the stream information from the log file.

    Parameters
    ----------
    logpath : str
        Path to the log file (txt file).

    Returns
    -------
    log : list
        List containing a list [temp, name, volt] for each stream.
        'temp' is the temperature in mK.
        'name' is the name of the stream.
        'volt' is the bias voltage in V.
    """
    log = list()
    with open(logpath) as log_file:
        for line in log_file:
            if 'mK' in line:
                temperature = int(*re.findall('(\d+) mK', line))
            if 'BIN0' in line:
                voltbias = float(re.findall('([0-9]*\.?[0-9]*)\s?V', line)[0])
                rload = float(*re.findall('RL = (\w+) Ohms', line))
                ibias = voltbias / rload
                filename = str(*re.findall(': (\w+).BIN0', line))
                log.append([temperature, filename, ibias])

    return log


def explore_plot(x_data, y_data,
                 num='EXPLORE PLOT', figsize=(11,7),
                 label='data', **kwargs):
    """Plots the given x_data and y_data according to the matplot
    keywords given. Also plot the welch psd of the y_data.

    Parameters
    ----------
    x_data : array_like
        Data in abscisse.
    y_data : array_like
        Data in ordinate, must be the same size as x_data.
    num : int or str, optional
        Use to select the figure where to plot the data.
    figsize : tuple of integers, optional, default: (6,4)
        width and height of figure
    Additionnal kwargs will be passed to the plot function.

    Returns
    -------
    figure : Figure

    Examples
    --------
    >>> time_array = np.arange(0, 1, 1e-3)
    >>> funk = lambda t: np.sin(t*13*2*np.pi)
    >>> explore_plot(time_array, funk(time_array),
                     num='test explore_plot')

    """
    fig = plt.figure(num=num, figsize=figsize)
    ax = fig.get_axes()
    # Check if the figure was already plotted.
    if not len(ax):
        fig, ax = plt.subplots(2, num=num, figsize=figsize)
    ax[0].plot(x_data, y_data, label=label, **kwargs)
    ax[0].set_xlabel('Time [$s$]')
    ax[0].set_ylabel('Signal')
    freq, psd = sgl.welch(y_data, fs=1e3, window='boxcar', nperseg=1e3)
    ax[1].plot(freq[1:], psd[1:], label=label, **kwargs)
    ax[1].set_xlabel('Frequency [$Hz$]')
    ax[1].set_ylabel('Signal PSD')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    for a in ax:
        a.legend(fontsize='small', ncol=1)
        a.grid(b=True)
    fig.tight_layout()
