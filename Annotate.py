#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of annotation routines for MatPlotLib

Created on Sat Oct 21 13:59:11 2017

@author: dan
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import datetime as dt
import pandas as pd
import numpy as np

def _Data2Axes(d, ax):
    """Convert from data space to axes space
    """
    axis_to_data = ax.transAxes + ax.transData.inverted()
    data_to_axis = axis_to_data.inverted()
    return data_to_axis.transform(d)

def _gfa():
    """ Utility function to get first axes on the figure.
    """
    fig = plt.gcf()
    return fig.get_axes()[0]

def MonthFmt(ax):
    """Set the y-axis to months with names between tick marks

    Parameters
    ----------
    ax : Matplotlib axes
        axes to be formatted.
    """
    # Set up axis formatting
    # Format codes are at:
    # https://docs.python.org/3/library
    #             /datetime.html#strftime-and-strptime-behavior
    monthly = mdates.MonthLocator()
    monthFmt = mdates.DateFormatter('%b')
    blankFmt = mdates.DateFormatter(' ')
    bimonthly = mdates.DayLocator(15)

    ax.yaxis.set_major_locator(monthly)
    ax.yaxis.set_minor_locator(bimonthly)
    ax.yaxis.set_minor_formatter(monthFmt)
    ax.yaxis.set_major_formatter(blankFmt)

    ax.tick_params(axis='y', which='minor', color=(0,0,0,0))

def AddYAxis(ax, month=False, pad=None, percent=None):
    """Adds y-axis that's a mirror of the y-axis on left.

    Parameters
    ----------
    ax : Axes
        Axes to be mirrored
    month : boolean default False
        Flag indicating that the axis has month labels, which must be handled
        differently.
    pad : int Default None
        Adjustment for location of labels. How far from the axis the right
        side of the label is since it is right-justified. If not supplied,
        a pretty good estimate is used.
    percent : int, float, or None default None
        Use if a percentate is desired on the right y-axis labels.
        ``percent=365`` to give percent of a year.

    Returns
    -------
    pad : float
        Value of actually pad used. This can help as an adjustment starting
        point if the estimated value is not right.
    Notes
    -----
    1. Use this function just before the final fig.show() command. Anywhere
    else causes problems with plotting to the wrong axes.

    2. Padding tested on a Retina MacPowerbook. It might be off if another
    system is used.
    """
    ax2 = ax.twinx()
    ax2.grid(False) # is sitting on top of lines
    ax2.set_yticks(ax.get_yticks())
    ax2.set_ylim(ax.get_ylim())
    yax = ax.get_yaxis()
    yax2 = ax2.get_yaxis()
    if month:
        yax2.set_major_locator(yax.get_major_locator())
        yax2.set_minor_locator(yax.get_minor_locator())
        yax2.set_major_formatter(yax.get_major_formatter())
        yax2.set_minor_formatter(yax.get_minor_formatter())
    elif percent:
        ts = ax2.get_yticks()
        ax2.set_yticklabels(['{0:.0%}'.format(t/percent) for t in ts])
    else:
        ts = ax2.get_yticklabels()
        [t.set_ha('right') for t in ts]
        if not pad:
            ax.figure.canvas.draw()
            rend = ax.figure.canvas.get_renderer()
            bboxes = yax.get_ticklabel_extents(rend)
            w = max(b.width for b in bboxes)
            # Value (2.8) tested with Mac Retina Display.
            # Might not work on other systems.
            if plt.get_backend().startswith('Mac'):
                pad = w/2.8 - 2
            else:
                pad = w/1.4 - 2
        yax2.set_tick_params(pad=pad)
    #fixes axis overlap in 'ggplot' style
    ax2.spines['right'].set_alpha(0)
    if month:
        ax2.tick_params(axis='y', which='minor', color=(0,0,0,0))
    else:
        ax.tick_params(axis='y', color=(0,0,0,0))
        ax2.tick_params(axis='y', color=(0,0,0,0))

    return ax2, pad

def Baseline(range):
    """Add baseline to current plot.

    Paramters
    ---------
    range : list
        list containing first and last date of baseline range
        (e.g. [1890, 1920]). Must be in data units, not index values.

    Note
    ----
    Must be used after all data is plotted, since the scale may change if an
    additional line is plotted, moving the other lines away from the baseline
    annotation.
    """
    ax = _gfa()
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

def Attribute(ha='right', va='bottom', author='Chart: @dan613',
              source='', date=''):
    """Add an attribute to current plot.

    Parameters
    ----------
    ha : str ['right' | 'left'] opt default 'right'
        Horizontal alignment of attribute
    va : str ['top' | 'bottom' | 'below'] opt default 'bottom'
        Vertical alignment of attribute. 'below' puts it below the x-axis.
    author : str default 'Chart: @dan613'
        Chart author name
    source : str opt
        Where data came from (should start with 'Data:'). Multi-line strings
        can be used (lines separated by newline), with space indents.
    date : str opt
        Date/year chart was made or data created. Usually just year.

    Note
    ----
    If using the va='below' option, it is up to the calling routine to make
    sure there is enough room under the x-axis.
    """
    ax = _gfa()
    size = 'medium'
    loc = {'top': .99,
           'bottom': .01,
           'left': .01,
           'right': .99}
    if va == 'below':
        va = 'top'
        loc['top'] = -.05
        size = 'small'
    text = ''
    if source != '':
        text = source + '\n'
    if date == '':
        d = dt.datetime.now()
        date = d.strftime('%d %b %Y')
    text = text + author + '   ' + date
    plt.text(loc[ha], loc[va],
            text,
            ha=ha,
            va=va,
            multialignment='left',
            fontsize = size,
            transform=ax.transAxes)

def AddRate(*args, ax=None, label='{:.2}°C/decade', mult=10):
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
    ax : matplotlib.Axes
        Axis to plot rate on. Uses first axes if not provided
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
        print('Unexpected input. Use:\n  AddRate(x,y) or\n  AddRate(s)\n'
              'where x and y are lists-like, and s is a pandas series.')
    if ax==None:
        ax = _gfa()
    c = np.polyfit(x, y, 1)            # fit a line to data
    rate = label.format(c[0]*mult)
    xx = [x[0],
          (x[-1]-x[0])*.33 + x[0],
          x[-1]]                       # endpoints of fitted line
    p = np.poly1d(c)                   # create polynomial
    yy = p(xx)                         # calculate y values of line
    ax.plot(xx, yy, 'k-', linewidth=2, alpha=0.75)
    if c[0] >=0:
        xyt = (0, -3)
        va = 'top'
    else:
        xyt = (0, 3)
        va = 'bottom'

    ax.annotate(rate, xy=(xx[1], yy[1]), xycoords='data',
                xytext=xyt, textcoords='offset points',
                size='larger', ha='left', va=va)

