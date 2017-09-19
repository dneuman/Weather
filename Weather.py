#!/usr/bin/env python
"""Routines to deal with bulk weather data from Environment Canada

   Routines:
       * Setup
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
"""

import time
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from scipy import linalg
import statsmodels.api as sm


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
    """Get a year's worth of data from Environment Canada site
       as a list of strings in CSV format. Default is Ottawa
       in current year"""
    # update with correct URL info
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
    """Get desired years from online database, and merge them into the
       supplied records list."""
    if (eYear is None):
        eYear = time.localtime().tm_year
    if (sYear is None):
        sYear = eYear
    for theYear in range(sYear, eYear+1):
        tf = GetData(theYear, city)
        df = tf.combine_first(df)
    return df

def RawToDF(city=0, header=25):
    """
    Process all data in a directory, returning a data frame. `path` is the full
    path to the station directory, and assumes data is in `.csv` format.

    Use if you have downloaded data from Environment Canada separately
    with wget.
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
    """
    Save weather data into a csv file
    """
    file = "/".join([basepath, stationName[city], "Data/complete.csv"])
    df.to_csv(file,
              float_format="% .1f")

def LoadDF(city=0):
    """
    Load weather data into a data frame
    """
    file = "/".join([basepath, stationName[city], "Data/complete.csv"])
    df = pd.read_csv(file,
                     index_col=0,
                     header=0,
                     dtype=dataTypes,
                     parse_dates=True)
    return df

# http://matplotlib.org/users/style_sheets.html
# matplotlib.style.available shows available styles
# matplotlib.style.library is a dictionary of available styles
# user styles can be placed in ~/.matplotlib/

def GetMonths(df, col, func=np.mean):
    """
    Take a panda dataframe (df) and return a new dataframe containing
    monthly values for the supplied column index.
    The value is aggregated using the supplied numpy function.
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
    """
    Take a panda dataframe containing daily weather values and return a
    new dataframe containing annual values for the supplied list of columns.
    The values are aggregated using the supplied numpy function.
    """
    labels = df.columns[cols]
    yr = df.pivot_table(values=list(labels),
                        index=['Year'],
                        aggfunc=func)
    return yr

def Lowess(data, f=2. / 3., pts=None, itn=3):
    """Lowess(s, f=2./3., pts=None, itn=3) -> yEst
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.

    data:   (pandas.Series) contains data points in the scatterplot. The
            function returns the estimated (smooth) values of y.
    f:      (float) the fraction of the data set to use for smoothing. A
            larger value for f will result in a smoother curve.
    pts:    (int) Optionally, the explicit number of data points to be used for
            smoothing can be passed through pts instead of f.
    itn:    (int) The number of robustifying iterations. The function will run
            faster with a smaller number of iterations.
    yEst:   (pandas.Series) The return value containing the smoothed data.
    """
    # Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
    #          Dan Neuman (conversion to Pandas series)
    # License: BSD (3-clause)

    x = np.array(data.index)
    y = (data.values)
    n = len(data)
    if pts is None:
        f = np.min([f, 1.0])
        r = int(np.ceil(f * n))
    else:  # allow use of number of points to determine smoothing
        r = int(np.min([pts, n]))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yEst = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(itn):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yEst[i] = beta[0] + beta[1] * x[i]

        residuals = y - yEst
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    return pd.Series(yEst, index=data.index, name='Trend')

def SMLowess(data, f=2. /3., pts=None, iter=3):
    x = np.array(data.index)
    y = data.values
    n = len(data)
    if pts is not None:
        f = pts / n
    est = sm.nonparametric.lowess(y, x, frac=f, it=iter)
    return pd.Series(est[:,1], index=est[:,0], name='Trend')


def WeightedMovingAverage(s, size, const=False):
    """
    Apply a weighted moving average on the supplied series
    using the supplied window. The window is
    partially applied at the beginning and end of the data to reduce
    artifacts.

    s:     Pandas series containing data to be averaged
    size:  integer indicating how wide a triangular window to use
    const: keep number of data points used constant at beginning and end.
           If True, window is increase, but clipped to `size` points centred
           on current point.

    Returns Pandas Series containing smoothed data
    """
    def SetLimits(i, hw):
        # i: current data location where window is centred
        # hw: half window width
        ds = max(0, (i-hw))         # data start
        de = min(dataLen-1, (i+hw)) # data end
        ws = hw - (i - ds)          # window start
        we = hw + (de - i)          # window end
        return ds, de, ws, we

    size += (size+1) % 2  # make odd
    window = np.hamming(size)
    window /= window.sum()  # normalize window
    a = pd.Series(np.convolve(s, window, mode='same'),
                  index=s.index,
                  name=s.name)
    dataLen = len(a)
    hw = int((size - 1)/2) # half window width
    # convolve has boundary effects when there is no overlap with the window
    # Begining and end of 'a' must be adjusted to compensate.
    # np.average() effectively scales the weights for the different sizes.
    if const: # keep data points used the same, but increase window size
        ds = 0
        de = size
        for i in range(hw+1):  # fix the start
            window = np.hamming(2*(de-i)+1)
            a.iloc[i] = np.average(s.iloc[ds:de], weights=window[-size:])
        ds = dataLen-size-1
        de = dataLen-1
        for i in range(dataLen-hw-1, dataLen):  # fix the end
            window = np.hamming(2*(i-ds)+1)
            a.iloc[i] = np.average(s.iloc[ds:de], weights=window[:size])
    else: # keep window size the same, but follow current point
        for i in range(hw+1):  # fix the start
            (ds, de, ws, we) = SetLimits(i, hw)
            a.iloc[i] = np.average(s.iloc[ds:de], weights=window[ws:we])
        for i in range(dataLen-hw-1, dataLen):  # fix the end
            (ds, de, ws, we) = SetLimits(i, hw)
            a.iloc[i] = np.average(s.iloc[ds:de], weights=window[ws:we])
    return a


def TempPlot(df, size=15, fignum=1, showmean=True, city=0,
             cols=[4, 6, 8],
             annotatePDO=False):
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
    plt.style.use('ggplot')
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
    yr = GetYears(df)
    yr = yr.drop(yr.index[0])
    hi = WeightedMovingAverage(yr.iloc[:, 0], size)
    mn = WeightedMovingAverage(yr.iloc[:, 1], size)
    lo = WeightedMovingAverage(yr.iloc[:, 2], size)
    if change:
        hi = hi - yr.iloc[:30, 0].mean()
        mn = mn - yr.iloc[:30, 1].mean()
        lo = lo - yr.iloc[:30, 2].mean()
    fig = plt.figure(fignum)
    fig.clear()
    plt.style.use('ggplot')
    plt.plot(hi, 'r-', alpha=0.5)
    plt.plot(mn, 'k-', alpha=0.5)
    plt.plot(lo, 'c-', alpha=0.5)
    plt.ylabel('Temperature Change from Baseline (°C)')
    plt.xlabel('Year')
    plt.title("Change in " + stationName[city] + "'s Annual Temperature")
    plt.legend(loc='upper left')
    plt.show()
    return

def CompareSmoothing(df, cols=[8],
                     size=15,
                     frac=2./3., pts=None, itn=3,
                     fignum=9, city=0):
    """
    Comparison between moving weighted average and lowess smoothing.
    df: daily records for a city
    cols: list of columns to use. Currently only uses first column supplied.
    size: size of moving average window
    frac: fraction of data to use for lowess window
    itn:  number of iterations to use for lowess
    """
    yf = GetYears(df, cols)
    yf = yf - yf.iloc[:30].mean()
    col = yf.columns[0]
    ma = WeightedMovingAverage(yf[col], size)
    #mc = WeightedMovingAverage(yf[col], size, const=True)
    lo = Lowess(yf[col], f=frac, pts=pts, iter=itn)
    #so = SMLowess(yf[col], f=frac, pts=pts, iter=itn)
    fig = plt.figure(fignum)
    fig.clear()
    plt.style.use('ggplot')
    plt.plot(yf[col], 'ko-', lw=1, alpha=0.2,
             label=(stationName[city]+' Annual Mean Temperature'))
    plt.plot(ma, 'b-', alpha=0.5, lw=2, label='Weighted Moving Average')
    #plt.plot(mc, 'r-', alpha=0.5, lw=2, label='WMA Constant Window')
    plt.plot(lo, 'g-', alpha=0.5, lw=2, label='Lowess')
    #plt.plot(so, 'c-', alpha=0.5, lw=2, label='SM Lowess')
    plt.title('Comparison between Weighted Moving Average and Lowess')
    plt.legend(loc='upper left')
    plt.ylabel('Temperature Change from Baseline (°C)')
    # Annotate chart
    bx = [yf.index[0], yf.index[0], yf.index[30], yf.index[30]]
    by = [-.3, -.4, -.4, -.3]
    plt.plot(bx, by, 'k-', linewidth=2, alpha=0.5)
    plt.text(bx[1], by[1]-0.15, 'Baseline', size='larger')
    boxt = ("Moving Average:\n"
           "  Weights: Triangular (Hamming)\n"
           "  Size: {0}\n"
           "Lowess:\n"
           "  Size: {1}\n"
           "  Iterations: {2}\n"
           "Chart: @Dan613")
    box = boxt.format(size, pts, itn)
    plt.text(1987, -1.9, box)
    plt.show()
    return

def ErrorPlot(df, size=31, cols=[8], fignum=10, city=0):
    """
    Showing standard deviation of temperature from trend.

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
    plt.style.use('ggplot')
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
    """
    Go through all data and plot every day where a weather record
    is made: max and min highs and lows, and record rain and snow.

    df: DataFrame containing Environment Canada data with standard columns.
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
    """
    Go through all data and plot every day where it rained or snowed.

    df: DataFrame containing Environment Canada data with standard columns.
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

    df: DataFrame containing Environment Canada daily data with standard columns.
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
    Plots a bar chart of days each year over 25 and 30 °C. Assumes df is a
    daily table.
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
    # fix x locations +- ind
    p1 = plt.bar(ind, hotc.iloc[:,0], width, color='red')
    # fix prob with rows being diff length due to years not in list
    p2 = plt.bar(ind, warmc.iloc[:,0], width, color='orange')
    plt.legend((p2[0], p1[0]), ('Days > 25°C', 'Days > 30°C'))
    plt.title("Warm and Hot Days for "+stationName[city])
    plt.show()
