#!/usr/bin/env python
"""Routines to deal with bulk weather data from Environment Canada

   Routines:

   * Setup:

   * Data Management:
       * Update - Add years to data
       * Save - Save consolidated data
       * GetFirstDay - returns index to first day of data
       * GetLastDay - Returns index to last day of data
   * Data Combining:
       * GetMonths - Return data grouped by months
       * GetYears - Return data grouped by years
   * Data Plotting:
       * StackPlot - Put several plots on one page
       * TempPlot - Temperature plot with optional annotations
       * TrendPlot - Plot multiple temperature trends (e.g. min, max)
       * ErrorPlot - Plot showing 1 and 2 std dev from trend line
       * RecordsPlot - Show all records on one graph
       * PrecipPlot - Show precipitation
       * SnowPlot - Show snowfall
       * HotDaysPlot - Show number of hot days per year
   * Miscellaneous:
       * CompareSmoothing - Show how Lowess and WMA compare for trends
       * CompareWeighting - Show how different weight windows compare

   Requires Python 3 (tested on 3.6.1, Anaconda distribution)
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import Smoothing as sm

pd.options.display.float_format = '{:.1f}'.format  # change print format
plt.style.use('ggplot')
# http://matplotlib.org/users/style_sheets.html
# matplotlib.style.available shows available styles
# matplotlib.style.library is a dictionary of available styles
# user styles can be placed in ~/.matplotlib/

basepath = '/Users/Dan/Documents/Weather/Stations/'

monthS = ['Yr ', 'Jan', 'Feb', 'Mar', 'Apr',
          'May', 'Jun', 'Jul', 'Aug',
          'Sep', 'Oct', 'Nov', 'Dec']
monthL = ['Year', 'January', 'February', 'March', 'April',
          'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']
monthN = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4,
          'May':5, 'Jun':6, 'Jul':7, 'Aug':8,
          'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12,
          'January':1, 'February':2, 'March':3, 'April':4,
          'May':5, 'June':6, 'July':7, 'August':8,
          'September':9, 'October':10, 'November':11, 'December':12}

class WxDF(pd.DataFrame):
    """Weather data management class

    Requires a table of cities (at least one) in the same directory as the
    calling program. The table must be a CSV file named 'Cities.csv' with the
    following headings:
        city, station, path

    Example
    -------
    ::
        df = WxDF(0)
        df

    loads the saved for the first city in the Cities.csv table. Typing `df` by
    itself in iPython will print a summary of the data.
    """
    _nonHeadRows = 25
    _dataTypes = { #0: np.datetime64,  # "Date/Time" (not used as it is index)
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
    _metadata = ['city','period','type','path','station','id']

    _cf = None  # city information

    def __init__(self, *args, **kwargs):
        """Initialize WxDF object

        Parameters
        ----------
        id : int (optional, default 0)
            Row in the Cities.csv table to use to initialize data
        df : DataFrame (optional)
            Use this dataframe if provided, otherwise load data from disk
            using the path found in Cities.csv
        """
        if WxDF._cf is None:
            WxDF._cf = pd.read_csv('Cities.csv',
                              header=0,
                              skipinitialspace=True,
                              index_col=False)
        if len(args) == 0 or type(args[0]) == int:
            if len(args) == 0: id = 0
            else: id = args[0]
            if len(args) == 2 and type(args[1]) == pd.DataFrame:
                df = args[1]
            else:
                df = pd.read_csv(basepath+WxDF._cf.path[id],
                                 index_col=0,
                                 header=0,
                                 dtype=WxDF._dataTypes,
                                 parse_dates=True)
            super(WxDF, self).__init__(df)
            self.city = WxDF._cf.city[id]
            self.path = WxDF._cf.path[id]
            self.station = WxDF._cf.station[id]
            self.id = id
            self.type = 'daily'
            self.period = 'daily'
        else:
            super(WxDF, self).__init__(*args, **kwargs)
            self.city = ''
            self.type = ''
            self.path = ''
            self.period = ''


    @property
    def _constructor(self):
        return WxDF

    def __str__(self):
        """Return a summary of the data
        """
        hMap = {'Data Quality':'Qual', 'Max Temp (°C)':'Max Temp',
                'Min Temp (°C)':'Min Temp', 'Mean Temp (°C)':'Mean Temp',
                'Total Precip (mm)':'Precip'}

        def GetLine(r):
            # hdgs and lbl are defined outside function
            if hasattr(self.index[0], 'year'):
                st = str(self.index[r].date()).ljust(10)
            else:
                st = str(self.index[r]).ljust(10)
            for i, c in enumerate(lbl):
                num = max(7, len(hdgs[i])+2)
                if type(self[c].iloc[r]) is np.float64:
                    st = st + '{:.1f}'.format(self[c].iloc[r]).rjust(num)
                else:
                    st = st + str(self[c].iloc[r]).rjust(num)
            return st

        hdgs = '    Undefined'
        if not hasattr(self, 'period'):
            num = min(3, len(self.columns))
            lbl = list(self.columns[0:num])
            hdgs = [l.rjust(max(7,len(l))) for l in lbl]
        elif self.period == 'daily' or len(self.columns)==26:
            full = list(hMap.keys())
            avail = list(self.columns)
            lbl = []
            for h in full:
                if h in avail:
                    lbl.append(h)
            hdgs = [hMap[h] for h in lbl]
            hdgs = [h.rjust(max(7,len(h))) for h in hdgs]
        elif self.period == 'monthly':
            cols = [0, 5, 11]
            lbl = list(self.columns[cols])
            hdgs = [l.rjust(max(7,len(l))) for l in lbl]
        elif self.period == 'annual':
            lbl = list(self.columns)
            hdgs = [l.rjust(max(7,len(l))) for l in lbl]
        last = self.GetLastDay()
        first = self.GetFirstDay()
        s = ''
        if hasattr(self, 'city') and self.city is not None:
            s = "City: {0}  Type: {1}\n".format(self.city, self.type)
        s = s + "Date        " + "  ".join(hdgs)
        if first > 0:
            s = '\n'.join([s, GetLine(0), '...', GetLine(first-1)])
        for i in range(first, first+5):
            s = '\n'.join([s, GetLine(i)])
        s = '\n'.join([s,'...'])
        num = min(len(self.index), last+2)
        for i in range(last-9, num):
            s = '\n'.join([s, GetLine(i)])
        s = '\n'.join([s, '[{}r x {}c]'.format(len(self.index),
                                               len(self.columns))])
        if hasattr(self, 'city'):
            if hasattr(self.index[first], 'year'):
                years = self.index[last].year - self.index[first].year
            else:
                years = self.index[last] - self.index[first]
            s = s + '  Years: ' + str(years+1)
        return s

    def Str(self):
        print(self.__str__)

    def _GetData(self, year=None, raw=False):
        """Get a year's worth of data from Environment Canada site.

        year: (opt) Year to retrieve. Defaults to current year.
        """
        if year is None:
            year = time.localtime().tm_year
        baseURL = ("http://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
                   "format=csv&stationID={stn}&Year={yr}&timeframe=2"
                   "&submit=Download+Data")
        url = baseURL.format(stn=self.station,
                             yr=year)
        if raw:
            df = pd.read_csv(url, skiprows=self._nonHeadRows,
                         index_col=0,
                         parse_dates=True,
                         na_values=['M','<31'])
            return df

        df = pd.read_csv(url, skiprows=self._nonHeadRows,
                         index_col=0,
                         parse_dates=True,
                         dtype=self._dataTypes,
                         na_values=['M','<31'])
        return WxDF(self.id, df)

    def Update(self, sYear=None, eYear=None):
        """Merge desired years from online database,

        df:    dataframe of daily data
        sYear: (opt) start year, defaults to year of last data point
        eYear: (opt) end year, defaults to current year
        """
        def Combine(orig, new):
            for row in new.index:
                orig.loc[row] = new.loc[row]

        if (eYear is None):
            eYear = time.localtime().tm_year
        if (sYear is None):
            sYear = self.index[self.GetLastDay()].year
        for theYear in range(sYear, eYear+1):
            nf = self._GetData(theYear)
            Combine(self, nf)

    def Save(self):
        """Save consolidated weather data into a .csv file

        df:   data frame of daily weather to save
        city: (opt) City to retrieve. Defaults to first city in list.

        Assumes data saveable at /basepath/city/Data/
        """
        # TODO: create directories if required
        file = "".join([basepath, self.path])
        self.to_csv(file,
              float_format="% .1f")

    def GetFirstDay(self):
        """Return index to first day with valid data.

        Parameters
        ----------
        df : dataframe containing daily data

        Returns
        -------
        Integer: df.iloc[i] will give data on first day.
        """
        col = min(4, len(self.columns)-1)
        i=-1
        for i in range(len(self.index)):
            if not np.isnan(self.iat[i,col]): break
        return i

    def GetLastDay(self):
        """Return index to last day with valid data.

        Parameters
        ----------
        df : dataframe containing daily data

        Returns
        -------
        Integer: df.iloc[i] will give data on last day.
        """
        col = min(4, len(self.columns)-1)
        i = len(self.index)-1
        for i in range(len(self.index)-1,-1,-1):
            if not np.isnan(self.iat[i,col]): break
        return i

    def GetMonths(self, col, func=np.mean):
        """Convert daily data to monthly data

        col:   column to be combined
        func:  (opt) function to use for combining. Defaults to mean (average)
        Returns dataframe with the grouped monthly data in each columns

        Only works for 1 column at a time due to extra complexity of multi-level
        axes when months are already columns.
        """

        label = self.columns[col]
        avgs = self.pivot_table(values=[label],
                              index=['Year'],
                              columns=['Month'],
                              aggfunc=func)
        avgs = avgs[label]  # turn into simple dataframe for simplicity
        avgs.rename(columns=monthS, inplace=True)
        avgs.city = self.city
        avgs.period = 'monthly'
        avgs.type = func.__name__ + ' ' + label
        return avgs

    def GetMonth(self, cols=[4, 6], month=None, func=np.mean):
        """Convert daily data to yearly data for a particular month

        """
        if month == 0:  # Return full year
            return self.GetYears(cols=cols, func=func)

        if month is None:
            month = time.localtime().tm_mon
        labels = list(self.columns[cols])
        yf = self.loc[lambda df: df.Month == month]
        mf = yf.pivot_table(values=labels,
                              index=['Year'],
                              aggfunc=func)
        return mf

    def GetYears(self, cols=[4, 6, 8], func=np.mean):
        """Convert daily data to yearly data.

        df:    dataframe containing daily data
        cols:  (opt) columns to be combined. Defaults to min, max, avg temps
        func:  (opt) function to use for combining. Defaults to mean (average)
        Returns dataframe with the grouped annual data
        """
        labels = self.columns[cols]
        yf = self.pivot_table(values=list(labels),
                            index=['Year'],
                            aggfunc=func)

        # Now ma
        cyr = self.Year[-1]  # get current year
        lyr = cyr-1
        dfc = self.loc[lambda df: df.Year == cyr]
        c = dfc.count()  # days in current year

        dfl = self.loc[lambda df: df.Year == lyr]
        dfl = dfl.iloc[:c[labels[0]]] # make number of days same as current year
        pl = dfl.pivot_table(values=list(labels),
                            index=['Year'],
                            aggfunc=func)
        # adjust current year by last year's change from current day to year end
        for i in range(len(labels)):
            yf.iloc[-1, i] = (yf.iloc[-2, i] - pl.iloc[0, i]) + yf.iloc[-1, i]

        yf.city = self.city
        yf.period = 'annual'
        yf.type = func.__name__
        return yf

def Mirror():
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.grid(False) # is sitting on top of lines
    ax2.set_yticks(ax.get_yticks())
    ax2.set_ylim(ax.get_ylim())


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

def TempPlot(df, cols=[4, 6, 8], size=21, showmean=True, fignum=1):
    """Plot indicated columns of data, with optional annotations"""

    yr = df.GetYears(cols=cols)
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
        a = sm.WeightedMovingAverage(s, size)
        baseline = s.loc[:1921].mean()
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
    plt.title("Change in " + df.city + "'s Annual Temperature")
    # Annotate chart
    plt.plot(bx, by, 'k-', linewidth=2, alpha=0.5)
    plt.text(bx[1], by[1]-0.15, 'Baseline', size='larger')
    plt.legend(loc=2)

    Mirror()

    fig.show()
    return


def TrendPlot(df, cols=[4, 6, 8], size=21, change=True, fignum=2):
    """Simple smoothed plots with optional baseline.
    """
    yf = df.GetYears(cols=cols)
    if change:
        for col in yf.columns:
            yf[col] = yf[col] - yf[col][:30].mean()
    ma = [sm.WeightedMovingAverage(yf[col], size) for col in yf.columns]
    fig = plt.figure(fignum)
    fig.clear()
    for i, y in enumerate(ma):
        plt.plot(y, '-', alpha=0.5, linewidth=2, label=yf.columns[i])
    plt.ylabel('Temperature Change from Baseline (°C)')
    plt.xlabel('Year')
    plt.title("Change in " + df.city + "'s Annual Temperature")
    plt.legend(loc='upper left')
    Mirror()
    fig.show()
    return


def ErrorPlot(df, size=31, cols=[8], fignum=3):
    """Show standard deviation of temperature from trend.

    df: DataFrame containing Environment Canada data with standard columns.
    cols: list of columns to use. Currently only uses first column supplied.
    size: size of moving average window
    """
    yf = df.GetYears(cols)
    yf = yf - yf.iloc[:30].mean()
    col = yf.columns[0]
    ma = sm.WeightedMovingAverage(yf[col], size)
    err = (ma - yf[col])**2
    std = err.mean()**0.5
    fig = plt.figure(fignum)
    fig.clear()
    plt.plot(yf[col], 'ko-', lw=1, alpha=0.2,
             label=(df.city+' '+col))
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
    plt.title("Change in " + df.city + "'s Annual Temperature")
    plt.show()


def RecordsPlot(df, use=[0,1,2,3,4,5], fignum=4):
    """Plot all records in daily data.

    df:     DataFrame containing daily data with standard columns.
    use:    List of records to show.
    fignum: (opt) Figure to use. Useful to keep multiple plots separated.
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
    props = [props[i] for i in use]
    # Create list of daily records. Use 2016 as reference year (leap year)
    r = pd.Series(index=pd.date_range(dt.date(2016, 1, 1),
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
    plt.title('Daily Weather Records for ' + df.city)
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
    plt.title('Records per Year for ' + df.city)
    plt.show()
    print('Done')
    return


def PrecipPlot(df, fignum=5):
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
    plt.title('Days with Precipitation in '+ df.city)
    plt.legend(numpoints=1,
               bbox_to_anchor=(1.07, 1), loc='upper right', borderaxespad=0.)
    plt.axis([1940, 2020, '20160101', '20161231'])
    plt.show()
    # Plot number of records per year in a stacked bar chart
    print('Done')
    return


def SnowPlot(df, fignum=6):
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
        a = sm.WeightedMovingAverage(s, 15)
        for i in a.index:
            a[i] = dStart + pd.Timedelta(days=int(a[i] - 0.5))
        plt.plot(a, 'c-', linewidth=4)
    plt.title('First and Last Snowfall for ' + df.city)
    plt.legend(numpoints=1,
               loc='center left')
    plt.axis([1940, 2020, '20160101', '20161231'])
    plt.show()
    print('Done')
    return

def HotDaysPlot(df, fignum=7):
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
    plt.title("Warm and Hot Days for "+df.city)
    plt.show()

def MonthRangePlot(nf, month=None, pad=False, combine=True, fignum=8):
        if month is None:
            month = time.localtime().tm_mon
        maxc = nf.columns[4] # max:0, min:1, avg:2
        minc = nf.columns[6]
        avgc = nf.columns[8]
        short = 5  # length of short weighting window
        long = 29  # length of long weighting window
        vlong = 61 # length of daily weighting window
        # just use year, month, max, min, avg temps
        df = nf[nf.columns[[0,1,4,6,8]]].copy()
        # Get rid of rows that have 'nan' values
        df.dropna(inplace=True)

        # Calculate the standard deviation directly.
        # Std dev = square root of mean error squared
        # std dev = mean((val - mean(val))**2)**.5
        #
        # Using a moving average for the avg/mean for each day gives a better
        # error value since the mean temperature is different at the beginning
        # of a month than at the end.
        af = df.copy()  # holds moving average amounts
        ef = df[df.columns[[0,1,2,3]]].copy() # holds error amounts on max, min
        sf = ef.copy()  # holds standard deviation amounts
        mx = ef.copy()  # max, highest value above mean
        mn = ef.copy()  # min, lowest value below mean
        # Get the daily average max, min, avg temps.
        for c in [maxc, minc, avgc]:
            af[c] = sm.WeightedMovingAverage(df[c], size=vlong, pad=False)
        # Get the error and standard deviation
        for c in [maxc, minc]:
            ef[c] = df[c] - af[c]
            sf[c] = sm.WeightedMovingAverage(ef[c]**2,
                          size=vlong, pad=False)**0.5

        if month != 0:
            # reduce to just correct month for desired frames
            sf = sf.loc[lambda df: df.Month == month]
            ef = ef.loc[lambda df: df.Month == month]
            af = af.loc[lambda df: df.Month == month]
        # Get the average over year for max, min and avg temps
        af = af.pivot_table(values=[maxc, minc, avgc],
                               index=['Year'], aggfunc=np.mean)
        sf = sf.pivot_table(values=[maxc, minc],
                               index=['Year'], aggfunc=np.mean)
        # this is max and min *error*, so must be added to mean to be
        # useful.
        mx = ef.pivot_table(values=[maxc, minc],
                               index=['Year'], aggfunc=np.max)
        mn = ef.pivot_table(values=[maxc, minc],
                               index=['Year'], aggfunc=np.min)
        for c in [maxc, minc]:
            mx[c] = sm.WeightedMovingAverage(mx[c], size=short, pad=False)
            mn[c] = sm.WeightedMovingAverage(mn[c], size=short, pad=False)
            sf[c] = sm.WeightedMovingAverage(sf[c], size=long, pad=False)
        for c in [maxc, minc, avgc]:
            af[c] = sm.WeightedMovingAverage(af[c], size=long, pad=False)

        umaxt = af[maxc] + sf[maxc]
        lmaxt = af[maxc] - sf[maxc]
        umint = af[minc] + sf[minc]
        lmint = af[minc] - sf[minc]
        maxmaxt = af[maxc] + mx[maxc]
        minmaxt = af[maxc] + mn[maxc]
        maxmint = af[minc] + mx[minc]
        minmint = af[minc] + mn[minc]
        index = af.index

        title = 'Temperature Range in '+df.city+' for '+monthL[month]
        fig = plt.figure(fignum)
        fig.clear()
        if not combine:
            plt.subplots_adjust(hspace=0.001, wspace=0.1,
                                left=0.08, right=0.92,
                                bottom=0.05, top=0.95)
            ax0 = plt.subplot(2, 1, 1)
            ax1 = plt.subplot(2, 1, 2, sharex=ax0)
            xt = ax0.get_xticklabels()
            plt.setp(xt, visible=False)
            fig.suptitle(title)
        else:
            ax0 = plt.subplot(1, 1, 1)
            ax1 = ax0
            plt.title(title)

        ax0.fill_between(index, maxmaxt, minmaxt,
                         color='red', alpha=0.10, label='Upper/Lower Highs')
        ax1.fill_between(index, maxmint, minmint,
                         color='blue', alpha=0.10, label='Upper/Lower Lows')
        ax0.fill_between(index, umaxt, lmaxt,
                         color='red', alpha=0.15, label='68% Range Highs')
        ax1.fill_between(index, umint, lmint,
                         color='blue', alpha=0.15, label='68% Range Lows')
        ax0.plot(af[maxc], 'r-', lw=2, alpha=0.5, label='Average Highs')
        ax1.plot(af[minc], 'b-', lw=2, alpha=0.5, label='Average Lows')
        if combine:
            ax0.plot(af[avgc], 'k-', lw=2, alpha=0.5, label='Average Daily')
        ax0.legend(loc='upper left')
        ax0.set_ylabel('Temperature °C')
        if not combine:
            ax1.legend(loc='upper left')
            ax1.set_ylabel('Temperature °C')
        plt.show()




def CompareWeighting(df, cols=[8], size=31, fignum=20):
    """Compare various weighting windows on real data.
    """
    yf = df.GetYears(cols)
    yf = yf - yf.iloc[:30].mean()
    col = yf.columns[0]
    y = yf[col]
    fig = plt.figure(fignum)
    fig.clear()
    plt.plot(y, 'ko-', lw=1, alpha=0.15,
             label=(df.city+' '+col))

    ma = sm.WeightedMovingAverage(y, size, winType=sm.Triangle)
    plt.plot(ma, '-', alpha=0.8, lw=1, label='Triangle')

    w = sm.Triangle(size, clip=0.8)
    ma = sm.WeightedMovingAverage(y, size, wts=w)
    plt.plot(ma, '-', alpha=0.8, lw=1, label='Clipped Triangle (0.5)')

    ma = sm.WeightedMovingAverage(y, size, winType=np.hamming)
    plt.plot(ma, '-', alpha=0.8, lw=1, label='Hamming')

    ma = sm.WeightedMovingAverage(y, size)
    plt.plot(ma, '-', alpha=0.8, lw=1, label='Hanning')

    ma = sm.WeightedMovingAverage(y, size, winType=np.blackman)
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
                     fignum=21, city=0):
    """Comparison between moving weighted average and lowess smoothing.

    df:    daily records for a city
    cols:  list of columns to use. Currently only uses first column supplied.
    size:  size of moving average window
    frac:  fraction of data to use for lowess window
    itn:   number of iterations to use for lowess
    order: order of the lowess polynomial
    lags:     number of time lags to use for SSA
    """
    yf = df.GetYears(cols)
    yf = yf - yf.iloc[:30].mean()
    col = yf.columns[0]
    y = yf[col]
    fig = plt.figure(fignum)
    fig.clear()
    plt.plot(y, 'ko-', lw=1, alpha=0.15,
             label=(df.city+' '+col))
    if pts==None:
        p = np.ceil(frac * len(y))
    else:
        p = pts

    ma = sm.WeightedMovingAverage(y, size)
    plt.plot(ma, 'b-', alpha=0.5, lw=2, label='Weighted Moving Average')

    #mc = WeightedMovingAverage(y, size, const=True)
    #plt.plot(mc, 'r-', alpha=0.5, lw=2, label='WMA Constant Window')

    lo = sm.Lowess(y, f=frac, pts=pts, itn=itn)
    plt.plot(lo, 'g-', alpha=0.5, lw=2, label='Lowess (linear)')

    lp = sm.Lowess(y, f=frac, pts=pts, itn=itn, order=order)
    plt.plot(lp, 'g.', alpha=0.5, lw=2, label='Lowess (polynomial)')

    ss = sm.SSA(y, lags, rtnRC=2)
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

if __name__=='__main__':
    df = WxDF()
