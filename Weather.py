#!/usr/bin/env python
"""Routines to deal with bulk weather data from Environment Canada

   Routines:

   * Setup:

   * Data Management:
       * GetData - Get raw data from Environment Canada
       * AddYears - Add years to data
       * RawToDF - Load raw data obtained by wget
       * SaveDF - Save consolidated data
       * LoadDF - Load consolidated data
   * Data Combining:
       * GetMonths - Return data grouped by months
       * GetYears - Return data grouped by years
   * Data Smoothing:
       * Lowess - Use locally weighted scatterplot smoothing
       * SMLowess - Use statsmodel version of LOWESS
       * WeightedMovingAverage - Use a triangular window for moving avg
       * SSA - Use Singular Spectrum Analysis
   * Data Plotting:
       * TempPlot - Temperature plot with optional annotations
       * TempRangePlot - Plot multiple temperatures (e.g. min, max)
       * ErrorPlot - Plot showing 1 and 2 std dev from trend line
       * RecordsPlot - Show all records on one graph
       * PrecipPlot - Show precipitation
       * SnowPlot - Show snowfall
       * HotDaysPlot - Show number of hot days per year
   * Miscellaneous:
       * CompareSmoothing - Show how Lowess and WMA compare for trends

   Requires Python 3 (tested on 3.6.1, Anaconda distribution)
"""

import time
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm

pd.options.display.float_format = '{:.1f}'.format  # change print format
plt.style.use('ggplot')
# http://matplotlib.org/users/style_sheets.html
# matplotlib.style.available shows available styles
# matplotlib.style.library is a dictionary of available styles
# user styles can be placed in ~/.matplotlib/

basepath = '/Users/Dan/Documents/Weather/Stations/'
stationID = [4333]  # Ottawa
stationName = ['Ottawa']
nonHeadRows = 25
dataTypes = { #0: np.datetime64,  # "Date/Time"
             1: int,      # "Year"
             2: int,      # "Month",
             3: int,      # "Day",
             4: str,      # "Data Quality",
             5: float,    # "Max Temp (°C)",
             6: str,      # "Max Temp Flag",
             7: float,    # "Min Temp (°C)",
             8: str,      # "Min Temp Flag",
             9: float,    # "Mean Temp (°C)",
             10: str,     # "Mean Temp Flag",
             11: float,   # "Heat Deg Days (°C)",
             12: str,     # "Heat Deg Days Flag",
             13: float,   # "Cool Deg Days (°C)",
             14: str,     # "Cool Deg Days Flag",
             15: float,   # "Total Rain (mm)",
             16: str,     # "Total Rain Flag",
             17: float,   # "Total Snow (cm)",
             18: str,     # "Total Snow Flag",
             19: float,   # "Total Precip (mm)",
             20: str,     # "Total Precip Flag",
             21: float,   # "Snow on Grnd (cm)",
             22: str,     # "Snow on Grnd Flag",
             23: float,   # "Dir of Max Gust (10s deg)",
             24: str,     # "Dir of Max Gust Flag",
             25: float,   # "Spd of Max Gust (km/h)",
             26: str      # "Spd of Max Gust Flag"
             }


def GetData(year=None, city=0):
    """Get a year's worth of data from Environment Canada site.

    year: (opt) Year to retrieve. Defaults to current year.
    city: (opt) City to retrieve. Defaults to first city in list.
    Returns Pandas data frame with daily data.
    """
    if year is None:
        year = time.localtime().tm_year
    baseURL = ("http://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
               "format=csv&stationID={stn}&Year={yr}&timeframe=2"
               "&submit=Download+Data")
    url = baseURL.format(stn=stationID[city],
                         yr=year)
    df = pd.read_csv(url, skiprows=nonHeadRows,
                     index_col=0,
                     parse_dates=True,
                     dtype=dataTypes,
                     na_values=['M','<31'])
    return df

def AddYears(df, sYear=None, eYear=None, city=0):
    """Merge desired years from online database,

    df:    dataframe of daily data
    sYear: (opt) start year, defaults to eYear
    eYear: (opt) end year, defaults to current year
    city:  (opt) City to retrieve. Defaults to first city in list.
    Returns updated dataframe
    """
    if (eYear is None):
        eYear = time.localtime().tm_year
    if (sYear is None):
        sYear = eYear
    for theYear in range(sYear, eYear+1):
        tf = GetData(theYear, city)
        df = tf.combine_first(df)
    return df

def RawToDF(city=0, header=25):
    """Process all data in a directory, returning a data frame.

    Use if you have downloaded data from Environment Canada separately
    with wget. Assumes data in .csv format.

    city:   (opt) City to retrieve. Defaults to first city in list.
    header: (opt) Number of header rows in each .csv file. Defaults to 25
    """
    template = "/".join([basepath, stationName[city],"Raw/*.csv"])
    first = True
    files = glob.glob(template)
    if len(files)==0:
        print("no files for: {0}".format(template))
        return
    for file in files:
        tf = pd.read_csv(file,
                         index_col=0,
                         dtype=dataTypes,
                         parse_dates=True,
                         na_values=['M','<31'],
                         skiprows=header)
        if first:
            df = tf
            first = False
        else:
            df = pd.concat([df, tf])
    return df

def SaveDF(df, city=0):
    """Save consolidated weather data into a .csv file

    df:   data frame of daily weather to save
    city: (opt) City to retrieve. Defaults to first city in list.

    Assumes data saveable at /basepath/city/Data/
    """
    file = "/".join([basepath, stationName[city], "Data/complete.csv"])
    df.to_csv(file,
              float_format="% .1f")

def LoadDF(city=0):
    """Load weather data into a data frame

    city: (opt) City to retrieve. Defaults to first city in list.

    Assumes data located at /basepath/city/Data/
    """
    file = "/".join([basepath, stationName[city], "Data/complete.csv"])
    df = pd.read_csv(file,
                     index_col=0,
                     header=0,
                     dtype=dataTypes,
                     parse_dates=True)
    return df


def GetMonths(df, col, func=np.mean):
    """Convert daily data to monthly data

    df:    dataframe containing daily data
    col:   column to be combined
    func:  (opt) function to use for combining. Defaults to mean (average)
    Returns dataframe with the grouped monthly data in each columns

    Only works for 1 column at a time due to extra complexity of multi-level
    axes when months are already columns.
    """

    colNames = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
                5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    label = df.columns[col]
    avgs = df.pivot_table(values=[label],
                          index=['Year'],
                          columns=['Month'],
                          aggfunc=func)
    avgs = avgs[label]  # turn into simple dataframe for simplicity
    avgs.rename(columns=colNames, inplace=True)
    return avgs


def GetYears(df, cols=[4, 6, 8], func=np.mean):
    """Convert daily data to yearly data.

    df:    dataframe containing daily data
    cols:  (opt) columns to be combined. Defaults to min, max, avg temps
    func:  (opt) function to use for combining. Defaults to mean (average)
    Returns dataframe with the grouped annual data
    """
    labels = df.columns[cols]
    yr = df.pivot_table(values=list(labels),
                        index=['Year'],
                        aggfunc=func)
    cyr = df.Year[-1]  # get current year
    lyr = cyr-1
    dfc = df.loc[lambda df: df.Year == cyr]
    c = dfc.count()

    dfl = df.loc[lambda df: df.Year == lyr]
    dfl = dfl.iloc[:c[labels[0]]] # make number of days same as current year
    pl = dfl.pivot_table(values=list(labels),
                        index=['Year'],
                        aggfunc=func)
    # adjust current year by last year's change from current day to year end
    for i in range(len(labels)):
        yr.iloc[-1, i] = (yr.iloc[-2, i] - pl.iloc[0, i]) + yr.iloc[-1, i]

    return yr

def Lowess(data, f=2./3., pts=None, itn=3, order=1, pad=True):
    """Fits a nonparametric regression curve to a scatterplot.

    Parameters
    ----------
    data : pandas.Series
        Data points in the scatterplot. The
        function returns the estimated (smooth) values of y.

    **Optionals**

    f : float
        The fraction of the data set to use for smoothing. A
        larger value for f will result in a smoother curve.
    pts : int
        The explicit number of data points to be used for
        smoothing instead of f.
    itn : int
        The number of robustifying iterations. The function will run
        faster with a smaller number of iterations.
    order : int
        The order of the polynomial used for fitting. Defaults to 1
        (straight line). Values < 1 are made 1. Larger values should be
        chosen based on shape of data (# of peaks and valleys + 1)
    pad : Boolean
        If True, will pad the data and the beginning and end by the amount
        of data used for smoothing (f*n or pts). This may provide better
        tracking of data at the ends.

    Returns
    -------
    pandas.Series containing the smoothed data.
    """
    # Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
    #            original
    #          Dan Neuman <https://github.com/dneuman>
    #            converted to Pandas series and extended to polynomials
    #            and added padding option.
    # License: BSD (3-clause)

    n = len(data)
    if pts is None:
        f = np.min([f, 1.0])
        r = int(np.ceil(f * n))
    else:  # allow use of number of points to determine smoothing
        r = int(np.min([pts, n]))
    r = min([r, n-1])
    order = max([1, order])
    if pad:
        tx = np.array(data.index, dtype=float)
        ty = data.values
        x = np.zeros(len(data)+2*r)
        y = np.zeros(len(data)+2*r)
        # pad data as a reflection of original data. eg use indexes:
        # 2, 1, 0, 1, 2, 3, 4, 5 and
        # n-3, n-2, n-1, n-2, n-3, n-4
        # x values keep the distances between them but go backwards at end and
        # forward at the end. y values are not changed.
        for i in range(r):  # pad beginning
            x[i] = tx[0] - (tx[r-i] - tx[0])
            y[i] = ty[r-i]
        for i in range(n):  # add actual data
            x[i+r] = tx[i]
            y[i+r] = ty[i]
        for i in range(r):  # pad end
            x[i+r+n] = tx[n-1] + (tx[n-1] - tx[n-2-i])
            y[i+r+n] = ty[n-2-i]
        n = len(y)
    else:
        x = np.array(data.index, dtype=float)
        y = data.values
    # condition x-values to be between 0 and 1 to reduce errors in linalg
    x = x - x.min()
    x = x / x.max()
    # Create matrix of 1, x, x**2, x**3, etc, by row
    xm = np.array([x**j for j in range(order+1)])
    # Create weight matrix, one column per data point
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    # Set up output
    yEst = np.zeros(n)
    delta = np.ones(n)  # Additional weights for iterations
    for iteration in range(itn):
        for i in range(n):
            weights = delta * w[:, i]
            xw = np.array([weights * x**j for j in range(order+1)])
            b = xw.dot(y)
            a = xw.dot(xm.T)
            beta = np.linalg.solve(a, b)
            yEst[i] = sum([beta[j] * x[i]**j for j in range(order+1)])
        # Set up weights to reduce effect of outlier points on next iteration
        residuals = y - yEst
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    if pad:
        n = len(data)
        return pd.Series(yEst[r:n+r], index=data.index, name='Trend')
    else:
        return pd.Series(yEst, index=data.index, name='Trend')

def SMLowess(data, f=2./3., pts=None, itn=3):
    """Return Statsmodels lowess smoothing as a series.

    data:  Data series containing data to be smoothed
    f:     Fraction of data to use for smoothing
    pts:   (opt) Number of data points to use instead of fraction
    itn:   (opt) number of iterations to smooth.
    Returns a data series containing smoothed points.
    """
    x = np.array(data.index)
    y = data.values
    n = len(data)
    if pts is not None:
        f = pts / n
    est = sm.nonparametric.lowess(y, x, frac=f, it=itn)
    return pd.Series(est[:,1], index=est[:,0], name='Trend')

def Triangle(size, clip=1.0):
    """Return a triangle weighting window, with optional clipping.

    Paramters
    ---------
    size : int
        Length of the returned weights

    **Optionals**

    clip : float
        Any weight above `clip` will be forced to `clip`.
    """
    w = np.bartlett(size+2)
    w = w[1:-1]  # remove zeros at endpoints
    w = np.array([min(clip, i) for i in w])
    return (w / max(w))

def Hanning(size):
    w = np.hanning(size+2)
    w = np.array(w[1:-1])  # remove zeros at endpoints
    return (w / max(w))

def WeightedMovingAverage(s, size, pad=True, winType=Hanning, wts=None):
    """Apply a weighted moving average on the supplied series.

    Parameters
    ----------
    s : pandas.Series
        data to be averaged
    size : integer
        how wide a triangular window to use

    **Optionals**

    pad : Boolean
        flag determining whether to pad beginning and end of data with a
        weighted average of the last `size` points. This provides better
        smoothing at the beginning and end of the line, but it tends to have
        zero slope.
    winType : Function
        Window function that takes an integer (window size) and returns a list
        of weights to be applied to the data. The default is Hanning, a
        weighted cosine with non-zero endpoints. Other possible windows are:

        * np.bartlett (triangular with endpoints of 0)
        * np.blackman (3 cosines creating taper)
        * np.hamming (weighted cosine)
        * np.hanning (weighted cosine with endpoints of 0)
    wts : list
        List of weights to use. `size` becomes the length of wts. Use this
        option to provide a custom weighting function. The length of wts
        should be odd, but this is not enforced.
    Returns
    -------
    Pandas Series containing smoothed data

    Notes
    -----
    Defaults to using a triangular window for weights, centered on
    each point. For points near the beginning or end of data, special
    processing is required that isn't in built-in functions.
    """
    def SetLimits(i, hw):
        # i: current data location where window is centred
        # hw: half window width
        ds = max(0, (i-hw))         # data start
        de = min(n-1, (i+hw)) # data end
        ws = hw - (i - ds)          # window start
        we = hw + (de - i)          # window end
        return ds, de, ws, we

    if type(wts) == type(None):
        size += (size+1) % 2  # make odd
        window = winType(size)
        window /= window.sum()  # normalize window
    else:
        window = wts / wts.sum()
        size = len(wts)
    n = len(s)
    hw = int(size / 2) # half window width
    # convolve has boundary effects when there is no overlap with the window
    # Begining and end of 'a' must be adjusted to compensate.
    # np.average() effectively scales the weights for the different sizes.
    if pad: # pad the data with reflected values
        # create padded beginning
        y = np.zeros(n+2*hw)
        for i in range(hw):
            y[i] = s.iloc[hw-i]
        for i in range(hw):
            y[i+n+hw] = s.iloc[n-i-1]
        for i in range(n):
            y[i+hw] = s.iloc[i]
        yc = np.convolve(y, window, mode='same')
        a = pd.Series(yc[hw:n+hw],
                      index=s.index,
                      name=s.name)
    else: # clip window as available data decreases
        a = pd.Series(np.convolve(s, window, mode='same'),
                      index=s.index,
                      name=s.name)
        for i in range(hw+1):  # fix the start
            (ds, de, ws, we) = SetLimits(i, hw)
            a.iloc[i] = np.average(s.iloc[ds:de], weights=window[ws:we])
        for i in range(n-hw-1, n):  # fix the end
            (ds, de, ws, we) = SetLimits(i, hw)
            a.iloc[i] = np.average(s.iloc[ds:de], weights=window[ws:we])
    return a

def SSA(s, m, rtnRC=1, pad=True):
    """Implement Singular Spectrum Analysis for pandas Series

    Parameters
    ----------
    s : pandas.Series
        Input data, in series or single columns of a data frame. Any necessary
        normalization (e.g. for anomolies from baseline) should be done.
    m : int
        Order or number of time lags to calculate over. A larger number gives
        more smoothing in the first returned column.

    **Optionals**

    rtnRC : int
        Number of reconstructed principles to return. Set to None to get all
        of them. Most smoothing is done in first returned column, but other
        columns may be useful to see periodicities.
    pad : Boolean
        Flad to pad data and principle components instead of using zeros.
        Reflected data is used, mirrored around the end point near the zeros.

    Returns
    -------
    pandas.DataFrame containing the reconstructed principles (or just the first
    one if allRC is True).

    Notes
    -----
    Computes the first m principle components (PCs) using Singular Spectrum
    Analysis. Most of the trend information is in the first reconstructed PC
    (RC), so the function returns just the first RC by default. This RC will
    look like smoothed data, and the amount of smoothing is determined by how
    large `m` is. Note that padding is added to prevent a drop towards 0 at
    beginning and end.

    Examples
    --------
    from:
    http://environnement.ens.fr/IMG/file/DavidPDF/SSA_beginners_guide_v9.pdf
    ::

        %precision 2
        import pandas as pd
        import numpy as np
        import matplotlib as plt

        y = [1.0135518, - 0.7113242, - 0.3906069, 1.565203, 0.0439317,
             - 1.1656093, 1.0701692, 1.0825379, - 1.2239744, - 0.0321446,
             1.1815997, - 1.4969448, - 0.7455299, 1.0973884, - 0.2188716,
             - 1.0719573, 0.9922009, 0.4374216, - 1.6880219, 0.2609807]

        rc = SSA(pd.Series(y), 4, allRC=None, pad=False)
        plt.plot(rc)
        plt.show()

        rc[0].values.flatten()

        array([ 0.3 , -0.31, -0.33,  0.82, -0.06, -0.82,  0.54,  0.53, -0.88,
                0.07,  0.83, -0.66, -0.36,  0.83, -0.18, -0.72,  0.63,  0.23,
               -0.68,  0.24])

    """
    n = len(s)
    y = np.array(s.values)
    mr = range(m)
    avg = y[int(n*0.9):].mean()  # use last 10% average to pad
    ys = np.ones((n,m)) * avg    # time shifted y-values
    for i in mr:
        ys[:n-i,i] = y[i:]
        if pad: ys[n-i:,i] = y[n-2:n-i-2:-1] # reflect end data
    # get autocorrelation at first `order` time lags
    cor = np.correlate(y, y, mode='full')[n-1:n-1+m]/n
    # make toeplitz matrix (diagonal, symmetric)
    c = np.array([[cor[abs(i-j)] for i in mr] for j in mr])
    # get eigenvalues and eigenvectors
    lam, rho = np.linalg.eig(c)
    pc = ys.dot(rho)  # principle components
    # reconstruct the components in proper time frame
    if rtnRC is None:
        desired = m
    else:
        desired = min(m, rtnRC)
    rc = np.zeros((n, desired))
    for j in range(desired):
        z = np.zeros((n, m))
        for i in mr:  # make time shifted principle component matrix
            z[i:,i] = pc[:n-i, j]
            if pad: z[:i,i] = pc[i:0:-1, j]  # reflect beginning data
        rc[:,j] = z.dot(rho[:, j]) / m
    return pd.DataFrame(rc, index=s.index)

def StackPlot(df, cols=2, title='', fignum=20):
    """Create a series of plots above each other, sharing x-axis labels.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing data to be plotted in separate columns. Column
        names will be used for labels.

    **Optionals:**

    cols : int
        Number of columns per figure to use.
    title : String
        Name to use on each figure. Page numbers are added.
    fignum : int
        Figure to start at. Useful if you want to keep older plots available.

    Notes
    -----
    Useful for plotting the SSA reconstructed principles to see which might be
    important.
    """
    rows = 4
    n = len(df.columns)
    if n <= rows: cols=1
    plots = rows * cols
    ax = list(range(plots))
    pages = int(np.ceil(n / plots))
    title = title + ' Page {}'
    for page in range(pages):
        fig = plt.figure(fignum+page, figsize=(14,10))
        fig.clear()
        fig.suptitle(title.format(page+1), fontsize=16, y=0.98)
        plt.subplots_adjust(hspace=0.001, wspace=0.1,
                            left=0.05, right=0.95,
                            bottom=0.05, top=0.95)
        end = min(plots, n - page * plots)
        for i in range(end):
            loc = page * plots + i
            r = int(i/plots)
            if r==0:
                ax[i] = plt.subplot(rows, cols, i+1)
            else:
                ax[i] = plt.subplot(rows, cols, i+1, sharex=ax[i%cols])
            if i < end-cols:  # hide ticks on all but last cols plots
                xt = ax[i].get_xticklabels()
                plt.setp(xt, visible=False)
            col = df.columns[loc]
            ax[i].plot(df[col], label=col)
            plt.legend(loc='upper left')
    plt.show()



def TempPlot(df, size=15, fignum=1, showmean=True, city=0,
             cols=[4, 6, 8],
             annotatePDO=False):
    """Plot indicated columns of data, with optional annotations"""

    yr = GetYears(df, cols=cols)
    styles = [['r-', 'ro-'], ['c-', 'co-'], ['k-', 'ko-']]
    # Set baseline annotation line
    bx = [yr.index[0], yr.index[0], yr.index[30], yr.index[30]]
    by = [-.3, -.4, -.4, -.3]
    px = [1988, 2000, 2010]
    py = [0, 0, 0]
    cols = yr.columns
    if not showmean:
        cols = cols[0:-2]  # remove last column (mean)
    fig = plt.figure(fignum)
    fig.clear()  # May have been used before
    for ci, col in enumerate(cols):
        s = yr[col]
        a = WeightedMovingAverage(s, size)
        baseline = s.iloc[0:30].mean()
        s = s - baseline
        a = a - baseline
        plt.plot(a, styles[ci][0], alpha=0.5, label='Trend', lw=5)
        plt.plot(s, styles[ci][1], alpha=0.2, lw=2)
        for i, x in enumerate(px):
            py[i] = max(py[i], a[x])
        # fit line to recent data
        sr = s.loc[1970:]  # get recent raw data
        c = np.polyfit(sr.index, sr, 1)     # fit a line to data
        rate = '{:.2}°C/decade'.format(c[0]*10)
        x = [sr.index[0], 0, sr.index[-1]]  # endpoints of fitted line
        x[1] = (x[0]+x[2])/2                # add halfway point for comment
        p = np.poly1d(c)                    # create polynomial
        y = p(x)                            # calculate y values of line
        plt.plot(x, y, 'k-', linewidth=2, alpha=0.5)
        plt.text(x[1]-5, y[1]-0.3, rate, size='larger')
    # Draw chart
    plt.ylabel('Temperature Change From Baseline (°C)')
    plt.xlabel('Year')
    plt.title("Change in " + stationName[city] + "'s Annual Temperature")
    # Annotate chart
    plt.plot(bx, by, 'k-', linewidth=2, alpha=0.5)
    plt.text(bx[1], by[1]-0.15, 'Baseline', size='larger')
    if annotatePDO:
        # Define PDO locations for annotations
        py = [a[px[0]]+.05, a[px[1]]+.05, a[px[2]]+.05]
        pt = [px[1]-5, py[1]+.4]
        psx = [pt[0]-2, pt[0], pt[0]+2]
        psy = [pt[1]-.05, pt[1]-.05, pt[1]-.05]
        # Add PDO annotation
        plt.text(1890, -1.9, 'PDO = Pacific Decadal Oscillation')
        plt.text(pt[0], pt[1], 'PDO', size='larger', ha='center')
        pdoprops = dict(facecolor='black', width=2, alpha=0.5)
        for i in range(3):
            plt.annotate("", xy=(px[i], py[i]),
                         xytext=(psx[i], psy[i]),
                         arrowprops=pdoprops)

    plt.legend(loc=2)
    plt.show()
    return


def TempRangePlot(df, col=[4, 6, 8], size=15, change=True, fignum=2, city=0):
    """Simple plot with optional baseline.

    Currently requires 3 columns to work (min, max, mean).
    """
    yr = GetYears(df)
    hi = WeightedMovingAverage(yr.iloc[:, 0], size)
    mn = WeightedMovingAverage(yr.iloc[:, 1], size)
    lo = WeightedMovingAverage(yr.iloc[:, 2], size)
    if change:
        hi = hi - yr.iloc[:30, 0].mean()
        mn = mn - yr.iloc[:30, 1].mean()
        lo = lo - yr.iloc[:30, 2].mean()
    fig = plt.figure(fignum)
    fig.clear()
    plt.plot(hi, 'r-', alpha=0.5)
    plt.plot(mn, 'k-', alpha=0.5)
    plt.plot(lo, 'c-', alpha=0.5)
    plt.ylabel('Temperature Change from Baseline (°C)')
    plt.xlabel('Year')
    plt.title("Change in " + stationName[city] + "'s Annual Temperature")
    plt.legend(loc='upper left')
    plt.show()
    return

def CompareWeighting(df, cols=[8], size=31, fignum=8, city=0):
    """Compare various weighting windows on real data.
    """
    yf = GetYears(df, cols)
    yf = yf - yf.iloc[:30].mean()
    col = yf.columns[0]
    y = yf[col]
    fig = plt.figure(fignum)
    fig.clear()
    plt.plot(y, 'ko-', lw=1, alpha=0.15,
             label=(stationName[city]+' '+col))

    ma = WeightedMovingAverage(y, size, winType=Triangle)
    plt.plot(ma, '-', alpha=0.8, lw=1, label='Triangle')

    w = Triangle(size, clip=0.8)
    ma = WeightedMovingAverage(y, size, wts=w)
    plt.plot(ma, '-', alpha=0.8, lw=1, label='Clipped Triangle (0.5)')

    ma = WeightedMovingAverage(y, size, winType=np.hamming)
    plt.plot(ma, '-', alpha=0.8, lw=1, label='Hamming')

    ma = WeightedMovingAverage(y, size)
    plt.plot(ma, '-', alpha=0.8, lw=1, label='Hanning')

    ma = WeightedMovingAverage(y, size, winType=np.blackman)
    plt.plot(ma, '-', alpha=0.8, lw=1, label='Blackman')

    plt.title('Comparison of Window Types for Moving Average')
    plt.legend(loc='upper left')
    plt.ylabel('Temperature Change from Baseline (°C)')
    # Annotate chart
    bx = [yf.index[0], yf.index[0], yf.index[30], yf.index[30]]
    by = [-.3, -.4, -.4, -.3]
    plt.plot(bx, by, 'k-', linewidth=2, alpha=0.5)
    plt.text(bx[1], by[1]-0.15, 'Baseline', size='larger')

    plt.show()

def CompareSmoothing(df, cols=[8],
                     size=31,
                     frac=2./3., pts=31, itn=3, order=2,
                     lags=31,
                     fignum=9, city=0):
    """Comparison between moving weighted average and lowess smoothing.

    df:    daily records for a city
    cols:  list of columns to use. Currently only uses first column supplied.
    size:  size of moving average window
    frac:  fraction of data to use for lowess window
    itn:   number of iterations to use for lowess
    order: order of the lowess polynomial
    lags:     number of time lags to use for SSA
    """
    yf = GetYears(df, cols)
    yf = yf - yf.iloc[:30].mean()
    col = yf.columns[0]
    y = yf[col]
    fig = plt.figure(fignum)
    fig.clear()
    plt.plot(y, 'ko-', lw=1, alpha=0.15,
             label=(stationName[city]+' '+col))
    if pts==None:
        p = np.ceil(frac * len(y))
    else:
        p = pts

    ma = WeightedMovingAverage(y, size)
    plt.plot(ma, 'b-', alpha=0.5, lw=2, label='Weighted Moving Average')

    #mc = WeightedMovingAverage(y, size, const=True)
    #plt.plot(mc, 'r-', alpha=0.5, lw=2, label='WMA Constant Window')

    lo = Lowess(y, f=frac, pts=pts, itn=itn)
    plt.plot(lo, 'g-', alpha=0.5, lw=2, label='Lowess (linear)')

    lp = Lowess(y, f=frac, pts=pts, itn=itn, order=order)
    plt.plot(lp, 'g.', alpha=0.5, lw=2, label='Lowess (polynomial)')

    ss = SSA(y, lags, rtnRC=2)
    plt.plot(ss.iloc[:,0], 'r-', alpha=0.5, lw=2, label='SSA')

    ss2 = ss.sum(axis=1)
    plt.plot(ss2, 'r.', alpha=0.5, lw=2, label='SSA, 2 components')

    #so = SMLowess(y, f=frac, pts=pts, iter=itn)
    #plt.plot(so, 'c-', alpha=0.5, lw=2, label='SM Lowess')

    plt.title('Comparison between Weighted Moving Average, Lowess, and SSA'
              ' - Padded')
    plt.legend(loc='upper left')
    plt.ylabel('Temperature Change from Baseline (°C)')
    # Annotate chart
    bx = [yf.index[0], yf.index[0], yf.index[30], yf.index[30]]
    by = [-.3, -.4, -.4, -.3]
    plt.plot(bx, by, 'k-', linewidth=2, alpha=0.5)
    plt.text(bx[1], by[1]-0.15, 'Baseline', size='larger')
    boxt = ("Moving Average:\n"
           "  Weights: Cosine (Hanning)\n"
           "  Size: {0}\n"
           "Lowess:\n"
           "  Size: {1}\n"
           "  Iterations: {2}\n"
           "  Polynomial Order: {3}\n"
           "Singular Spectrum Analysis:\n"
           "  Lags: {4}\n"
           "Chart: @Dan613")
    box = boxt.format(size, p, itn, order, lags)
    plt.text(1987, np.floor(y.min())+.05, box)
    plt.show()
    return

def ErrorPlot(df, size=31, cols=[8], fignum=10, city=0):
    """Show standard deviation of temperature from trend.

    df: DataFrame containing Environment Canada data with standard columns.
    cols: list of columns to use. Currently only uses first column supplied.
    size: size of moving average window
    """
    yf = GetYears(df, cols)
    yf = yf - yf.iloc[:30].mean()
    col = yf.columns[0]
    ma = WeightedMovingAverage(yf[col], size)
    err = (ma - yf[col])**2
    std = err.mean()**0.5
    fig = plt.figure(fignum)
    fig.clear()
    plt.plot(yf[col], 'ko-', lw=1, alpha=0.2,
             label=(stationName[city]+' '+col))
    plt.plot(ma, 'r-', alpha=0.5, lw=2, label='Weighted Moving Average')
    plt.fill_between(ma.index, ma.values+std, ma.values-std,
                     color='red', alpha=0.15, label='68%')
    plt.fill_between(ma.index, ma.values+2*std, ma.values-2*std,
                     color='red', alpha=0.10, label='95%')
    plt.legend(loc='upper left')
    # Annotate chart
    bx = [yf.index[0], yf.index[0], yf.index[30], yf.index[30]]
    by = [-.3, -.4, -.4, -.3]
    plt.plot(bx, by, 'k-', linewidth=2, alpha=0.5)
    plt.text(bx[1], by[1]-0.15, 'Baseline', size='larger')
    plt.ylabel('Temperature Change from Baseline (°C)')
    plt.title("Change in " + stationName[city] + "'s Annual Temperature")
    plt.show()


def RecordsPlot(df, fignum=5, city=0):
    """Plot all records in daily data.

    df:     DataFrame containing daily data with standard columns.
    fignum: (opt) Figure to use. Useful to keep multiple plots separated.
    city:   (opt) City to use for titles. Defaults to first city in list.
    """

    def ToNow(t):
        """Take a timestamp and return same day in 2016"""
        return pd.Timestamp(dt.date(2016, t.month, t.day))

    # set up data for each set of records:
    # [Name, df column, mark color and format, zorder]
    props = [
             ['Max Day', 4, 'r^', 6, float.__gt__, -100.0],
             ['Min Day', 4, 'rv', 5, float.__lt__, 100.0],
             ['Max Night', 6, 'b^', 4, float.__gt__, -100.0],
             ['Min Night', 6, 'bv', 3, float.__lt__, 100.0],
             ['Rain', 14, 'go', 2, float.__gt__, -100.0],
             ['Snow', 16, 'cH', 1, float.__gt__, -100.0],
             ]
    # Create list of daily records. Use 2016 as reference year (leap year)
    r = pd.TimeSeries(index=pd.date_range(dt.date(2016, 1, 1),
                                          dt.date(2016, 12, 31)))
    # Create list of counts of records for each year
    cols = []
    [cols.append(p[0]) for p in props]
    counts = pd.DataFrame(columns=cols,
                          index=list(range(df.index[0].year,
                                           df.index[-1].year+1)))
    counts.iloc[:, :] = 0
    fig = plt.figure(fignum, figsize=(17, 9))
    fig.clear()
    for p in props:
        print(p[0])
        x = []
        y = []
        # choose appropriate comparison function. The 'Min' records are '<'
        compare = p[4]
        r[:] = p[5]
        for i in range(len(df.index)):
            s = df.iat[i, p[1]]
            date = df.index[i]
            ndate = ToNow(date)
            if compare(s, r[ndate]):
                r[ndate] = s
                x.append(df.index[i].year)
                y.append(ndate)
                counts[p[0]].loc[date.year] += 1
        plt.plot(x, y,
                 p[2],
                 label=p[0], zorder=p[3])
    # Plot records data
    plt.title('Daily Weather Records for ' + stationName)
    plt.legend(numpoints=1,
               bbox_to_anchor=(1.07, 1), loc='upper right', borderaxespad=0.)
    plt.axis([1940, 2020, '20160101', '20161231'])
    plt.show()
    # Plot number of records per year in a stacked bar chart
    x = list(counts.index)
    y = list(range(len(counts.columns)))
    for i, v in enumerate(counts.columns):
        y[i] = list(counts[v])
    fig = plt.figure(fignum+1, figsize=(17, 9))
    fig.clear()
    plt.axis([1940, 2020, 0, 50])
    plt.stackplot(x, y)
    plt.legend(counts.columns)
    plt.title('Records per Year for ' + stationName[city])
    plt.show()
    print('Done')
    return


def PrecipPlot(df, fignum=6, city=0):
    """Go through all data and plot every day where it rained or snowed.

    df:     DataFrame containing Environment Canada data with standard columns.
    fignum: (opt) Figure to use. Useful to keep multiple plots separated.
    city:   (opt) City to use for titles. Defaults to first city in list.
    """

    def ToNow(t):
        """Take a timestamp and return same day in 2016"""
        return pd.Timestamp(dt.date(2016, t.month, t.day))

    # set up data for each set of records:
    # [Name, df column, mark color and format, zorder]
    props = [
             ['Rain', 14, 'g.', 2],
             ['Snow', 16, 'cH', 1],
             ]
    fig = plt.figure(fignum, figsize=(17, 9))
    fig.clear()
    for p in props:
        print(p[0])
        x = []
        y = []
        for i in range(len(df.index)):
            s = df.iat[i, p[1]]  # get sample
            date = df.index[i]
            ndate = ToNow(date)
            if s > 0:  # was there precipitation?
                x.append(df.index[i].year)
                y.append(ndate)
        plt.plot(x, y,
                 p[2],
                 label=p[0], zorder=p[3])
    # Plot records data
    plt.title('Days with Precipitation in '+ stationName[city])
    plt.legend(numpoints=1,
               bbox_to_anchor=(1.07, 1), loc='upper right', borderaxespad=0.)
    plt.axis([1940, 2020, '20160101', '20161231'])
    plt.show()
    # Plot number of records per year in a stacked bar chart
    print('Done')
    return


def SnowPlot(df, fignum=7, city=0):
    """
    Go through all data and plot first and last day of snow for the year.

    df:     DataFrame containing daily data with standard columns.
    fignum: (opt) Figure to use. Useful to keep multiple plots separated.
    city:   (opt) City to use for titles. Defaults to first city in list.
    """

    # set up data for each set of records:
    # [Name, df column, mark color and format, zorder]
    props = [
             ['First', 16, 'cH-', 1, pd.Timestamp.__lt__],
             ['Last', 16, 'co-', 2, pd.Timestamp.__gt__]
             ]
    # Create list of daily records. Use 2016 as reference year (leap year)
    cols = []
    [cols.append(p[0]) for p in props]
    r = pd.DataFrame(columns=cols,
                     index=list(range(df.index[0].year,
                                      df.index[-1].year+1)))
    dStart = pd.Timestamp(dt.date(2016, 1, 1))
    dMid = pd.Timestamp(dt.date(2016, 7, 1))
    dEnd = pd.Timestamp(dt.date(2016, 12, 31))
    r['First'] = dEnd
    r['Last'] = dStart
    fig = plt.figure(fignum, figsize=(17, 9))
    fig.clear()
    for p in props:
        col = p[0]
        print(col)
        compare = p[4]  # which comparison function to use
        for i, date in enumerate(df.index):
            s = df.iat[i, p[1]]
            if s == 0:  # only worry about dates with precipitation
                continue
            # put date into 2016 for plotting purposes
            pdate = pd.Timestamp(dt.date(2016, date.month, date.day))
            if compare(pdate, dMid):  # skip if wrong part of year
                continue
            yr = date.year
            if compare(pdate, pd.Timestamp(r.loc[yr, col])):
                r.loc[yr, col] = pdate
        plt.plot(list(r.index),
                 list(r[col]),
                 p[2],
                 linewidth=2,
                 alpha=0.5,
                 label=col)
        # Convert dates to day of year, get moving average, convert back
        # to actual date, then plot
        s = pd.Series(index=r.index,
                      data=[float(d.dayofyear) for d in r[col]])
        a = WeightedMovingAverage(s, 15)
        for i in a.index:
            a[i] = dStart + pd.Timedelta(days=int(a[i] - 0.5))
        plt.plot(a, 'c-', linewidth=4)
    plt.title('First and Last Snowfall for ' + stationName[city])
    plt.legend(numpoints=1,
               loc='center left')
    plt.axis([1940, 2020, '20160101', '20161231'])
    plt.show()
    print('Done')
    return

def HotDaysPlot(df, city, fignum=8):
    """
    Plots a bar chart of days each year over 25 and 30 °C.

    df:     DataFrame containing daily data with standard columns.
    fignum: (opt) Figure to use. Useful to keep multiple plots separated.
    city:   (opt) City to use for titles. Defaults to first city in list.
    """
    width = 0.35
    days = df.iloc[:, [0,4]]
    label = days.columns[1]
    hot = days[(days[label]>=30)]
    warm = days[(days[label]>=25)]
    hotc = hot.pivot_table(values=[label],
                           index=['Year'],
                           aggfunc=np.count_nonzero)
    warmc = warm.pivot_table(values=[label],
                             index=['Year'],
                             aggfunc=np.count_nonzero)
    fig = plt.figure(fignum)
    fig.clear()
    ind = np.arange(len(hotc.index))
    # TODO fix x locations +- ind
    p1 = plt.bar(ind, hotc.iloc[:,0], width, color='red')
    # TODO fix prob with rows being diff length due to years not in list
    p2 = plt.bar(ind, warmc.iloc[:,0], width, color='orange')
    plt.legend((p2[0], p1[0]), ('Days > 25°C', 'Days > 30°C'))
    plt.title("Warm and Hot Days for "+stationName[city])
    plt.show()
