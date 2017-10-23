#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of annotation routines for MatPlotLib

Created on Sat Oct 21 13:59:11 2017

@author: dan
"""

import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np

def _Data2Axes(d, ax):
    """Convert from data space to axes space
    """
    axis_to_data = ax.transAxes + ax.transData.inverted()
    data_to_axis = axis_to_data.inverted()
    return data_to_axis.transform(d)


def AddYAxis(pad=15):
    """Adds y-axis that's a mirror of the y-axis on left.

    Parameters
    ----------
    pad : int opt Default 15
        Adjustment for location of labels. How far from the axis the right
        side of the label is since it is right-justified.
    Notes
    -----
    Use this function just before the final fig.show() command. Anywhere else
    causes problems with plotting to the wrong axes.
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
    # make ticks invisible (not labels)
    ax.tick_params(axis='y', color=(0,0,0,0))
    ax2.tick_params(axis='y', color=(0,0,0,0))

def Baseline(range):
    """Add baseline to current plot.

    Paramters
    ---------
    range : list
        list containing first and last date of baseline range
        (e.g. [1890, 1920]). Must be in data units, not index values.
    """
    ax = plt.gca()
    if type(range) != list:
        print('Input to Baseline must be a list, eg [1850, 1920]')
        return
    bl = _Data2Axes((range[0], 0), ax)
    tr = _Data2Axes((range[1], 0), ax)
    btop = bl[1] - .05
    bbtm = bl[1] - .08
    blft = bl[0]
    brgt = tr[0]
    tr[1] -= .05
    bx = [blft, blft, brgt, brgt]
    by = [btop, bbtm, bbtm, btop]
    plt.plot(bx, by, 'k-', linewidth=1.5, alpha=0.75,
             transform=ax.transAxes)
    plt.text(bx[1], by[1]-.01, 'Baseline', size='larger',
             va='top', transform=ax.transAxes)

def Attribute(ha='right', va='bottom', source='', date=''):
    """Add an attribute to current plot.

    Parameters
    ----------
    ha : str ['right' | 'left'] opt default 'right'
        Horizontal alignment of attribute
    va : str ['top' | 'bottom'] opt default 'bottom'
        Vertical alignment of attribute
    source : str opt
        Where data came from (shoult start with 'Data:'). Multi-line strings
        can be used (lines separated by '\n'), with space indents.
    date : str opt
        Date/year chart was made or data created. Usually just year.
    """
    ax = plt.gca()
    loc = {'top': .86,
           'bottom': .12,
           'left': .14,
           'right': .89}
    text = ''
    if source != '':
        text = source + '\n'
    if date == '':
        d = dt.datetime.now()
        date = d.strftime('%d %b %Y')
    text = text + 'Chart: @dan613   ' + date
    plt.figtext(loc[ha], loc[va],
            text,
            ha=ha,
            va=va,
            multialignment='left',
            transform=ax.transAxes)

def AddRate(*args, label='{:.2}°C/decade', mult=10):
    """Add a rate line (linear regression) to current plot.

    AddRate(x, y, label='{:.2}°C/decade', mult=10) or
    AddRate(s, label='{:.2}°C/decade', mult=10)

    Parameters
    ----------
    x : array-like
        list of x-values
    y : array-like
        list of y-values
    s : pandas.Series
        Contains values and index instead of x, y
    label : str opt default '{:.2}°C/decade'
        Template for the rate label of the fitted line. Uses the str.format()
        values for a single variable imput.
    mult : float or int opt default 10
        How much to multiply the rate value for the label
    """
    if len(args)==1 and isinstance(args[0], pd.Series):
        x = args[0].index
        y = args[0].values
    elif len(args)==2:
        x = args[0]
        y = args[1]
    else:
        print('Unexpeced input. Use:\n  AddRate(x,y) or\n  AddRate(s)\n'
              'where x and y are lists-like, and s is a pandas series.')
    ax = plt.gca()
    c = np.polyfit(x, y, 1)            # fit a line to data
    rate = label.format(c[0]*mult)
    xx = [x[0],
          (x[-1]-x[0])*.33 + x[0],
          x[-1]]                       # endpoints of fitted line
    p = np.poly1d(c)                   # create polynomial
    yy = p(xx)                         # calculate y values of line
    plt.plot(xx, yy, 'k-', linewidth=2, alpha=0.75)
    lx, ly = _Data2Axes((xx[1], yy[1]), ax)
    plt.text(lx, ly-.01, rate, size='larger',
             ha='left', va='top', transform=ax.transAxes)
