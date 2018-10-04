#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Writes or read txt file to save results.

Author:
    Dimitri Misiak (misiak@ipnl.in2p3.fr)
"""


from os import path
from datetime import datetime


SIGN = ' =\n'


def savetxt(labels, values, fpath=None, header=''):
    """ Save values for the corresponding labels in txt file.
    The default header contains the date of creation/last edit
    of the file.

    Parameters
    ----------
    labels : sequence
        Labels of the entries in the txt file.
    values : sequence
        Values of the entries. Size must match the labels size.
    fpath : None or path, optional, default=None
        Absolute or relative path of the save txt file. By default, the save
        file is created in the current working directory with its name being
        the date.
    header : str, optional
        Additional info to be put into the header. Do not use the signature
        string in this header, this prevents the correct loading
        of the save file.

    See also
    --------
    loadtxt
    """
    # get the current date
    now = datetime.now()
    date = now.strftime("%Y-%m-%d %H:%M")

    # default name
    if fpath == None:
        fpath = 'save_{}'.format(date)

    # creates the lines of the save file
    entries = ['{0}{2}{1}\n\n'.format(l,v,SIGN) for l,v in zip(labels, values)]

    # writes in the save file the header and the entries
    with open(fpath, 'w') as save:
        save.writelines( date + '\n{}\n\n'.format(header) )
        save.writelines(entries)


def loadtxt(fname):
    """ Load a save txt file created with the function savetxt.

    Parameters
    ----------
    fname : path
        Absolute or relative path of the save file.

    Returns
    -------
    labels : sequence of str
        Labels of the save file.
    values : sequence of str
        Values corresponding to the labels, in string format.

    See also
    --------
    savetxt
    """
    sign = ' =\n'
    labels = list()
    values = list()

    with open(fname, 'r') as save:
        # read file until the end
        while 1:

            line = save.readline()

            if sign in line:
                # gather the labels and values in the file
                labels.append( line.replace(SIGN, '') )
                line = save.readline()
                values.append( line.replace('\n', '') )

            # end of the file
            if line == '':
                break

    return labels, values


if __name__ == '__main__':

    from general import choose_dir

    SAVEPATH = path.join(choose_dir(), 'savesys test')
    LABELS = ('A', 'B', 'C', 'D', 'E')
    VALUES = (1,2,3,4,5)

    savetxt(LABELS, VALUES, fpath=SAVEPATH, header='This is a test file !')

    lab, val = loadtxt(SAVEPATH)

    print lab
    print val
