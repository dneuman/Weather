#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of annotation routines for MatPlotLib

Created on Sat Oct 21 13:59:11 2017

@author: dan
"""

import matplotlib.pyplot as plt
import time

def AddYAxis(pad=20):
    """Adds y-axis that's a mirror of the y-axis on left.

    Parameters
    ----------
    prec : int (opt) Default=1
        Number of significant digits (precision) to use on scale
    Notes
    -----
    Use this function just before the final fig.show() command.
    """

    """
    Alternate approach:
        fmt = '{' + 'x: .{p}f'.format(p=1) + '}' # adds a space if number is >0
        ax.yaxis.set_major_formatter(tk.StrMethodFormatter(fmt))
        ax2.yaxis.set_major_formatter(tk.StrMethodFormatter(fmt))
    This approach uses the ASCII hyphen instead of unicode minus (\u2212).
    The downside is that the padding is not always the same as a hyphen.
    """
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.grid(False) # is sitting on top of lines
    ax2.set_yticks(ax.get_yticks())
    ax2.set_ylim(ax.get_ylim())
    ts = ax2.get_yticklabels()
    [t.set_ha('right') for t in ts]
    yax = ax2.get_yaxis()
    # Best padding depends on size of labels (length and font size)
    # get_window_extent().width shows this, but having trouble getting values.
    yax.set_tick_params(pad=pad)

def Baseline(range):
    """Add baseline to current plot.

    Paramters
    ---------
    range : list
        list containing first and last date of baseline range
        (e.g. [1890, 1920]). Must be in data units, not index values.
    """

    bx = [range[0], range[0], range[1], range[1]]
    by = [-.3, -.4, -.4, -.3]
    plt.plot(bx, by, 'k-', linewidth=2, alpha=0.75)
    plt.text(bx[1], by[1]-0.155, 'Baseline', size='larger')

def Attribute(h='right', v='bottom', source='', date=''):
    """Add an attribute to current plot.

    Parameters
    ----------
    h : str ['right' | 'left'] opt default 'right'
        Horizontal alignment of attribute
    v : str ['top' | 'bottom'] opt default 'bottom'
        Vertical alignment of attribute
    source : str opt
        Where data came from (shoult start with 'Data:'). Multi-line strings
        can be used (lines separated by '\n'), with space indents.
    date : str opt
        Date/year chart was made or data created. Usually just year.
    """
    ax = plt.gca()
    loc = {'top': .9,
           'bottom': .1,
           'left': .1,
           'right': .9}
    text = ''
    if source != '':
        text = source + '\n'
    if date == '':
        date = str(time.localtime().tm_year)
    text = text + 'Chart: @dan613  ' + date
    plt.figtext(loc[h], loc[v],
            text,
            ha=h,
            va=v,
            multialignment='left',
            transform=ax.transAxes)
