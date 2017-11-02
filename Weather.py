#!/usr/bin/env python
"""
Weather Module
**************

Routines to deal with bulk weather data from Environment Canada

WxDF Class
----------
   * Update - Add years to data
   * Save - Save consolidated data
   * GetFirstDay - returns index to first day of data
   * GetLastDay - Returns index to last day of data
   * GetMonths - Return data grouped by months
   * GetMonth - Return data grouped by years for a single month
   * GetYears - Return data grouped by years

Data Plotting
-------------
   * GridPlot - Put several plots on one page
   * TempPlot - Temperature plot with optional annotations
   * TrendPlot - Plot multiple temperature trends (e.g. min, max)
   * ErrorPlot - Plot showing 1 and 2 std dev from trend line
   * RecordsPlot - Show all records on one graph
   * PrecipPlot - Show precipitation
   * SnowPlot - Show snowfall
   * HotDaysPlot - Show number of hot days per year

Miscellaneous
-------------
   * CompareSmoothing - Show how Lowess and WMA compare for trends
   * CompareWeighting - Show how different weight windows compare

Requirements
------------
   Requires Python 3 (tested on 3.6.1, Anaconda distribution)
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import datetime as dt
import Smoothing as sm
import Annotate as at
import pathlib


pd.options.display.float_format = '{:.1f}'.format  # change print format
plt.style.use('weather')
# http://matplotlib.org/users/style_sheets.html
# matplotlib.style.available shows available styles
# matplotlib.style.library is a dictionary of available styles
# user styles can be placed in ~/.matplotlib/

class Settings():
    """Simple class to hold settings, making out of scope variables more
       obvious. Use str(obj) to get list of settings.
    """
    basepath = '/Users/Dan/Documents/Weather/Stations/'
    source = "Data: Environment Canada"
    tlw = 4  # trend linewidth
    dlw = 1  # data linewidth
    ta = 0.99   # trend alpha
    da = 0.3   # data alpha
    sa = 0.15  # std dev alpha
    ma = 0.1   # max/min alpha

    colors = {'doc':'Color from cycle to use per column',
              'Max Temp (°C)':'C0', 4:'C0',
              'Min Temp (°C)':'C1', 6:'C1',
              'Mean Temp (°C)':'C2', 8:'C2'} # colors to use per column
    monthS = {'doc':'Return short month name',
              0:'Yr ', 1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr',
              5:'May', 6:'Jun', 7:'Jul', 8:'Aug',
              9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    monthL = {'doc':'Return long month name',
              0:'Year', 1:'January', 2:'February', 3:'March', 4:'April',
              5:'May', 6:'June', 7:'July', 8:'August',
              9:'September', 10:'October', 11:'November', 12:'December'}
    monthN = {'doc':'Return index for supplied long or short month name',
              'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4,
              'May':5, 'Jun':6, 'Jul':7, 'Aug':8,
              'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12,
              'January':1, 'February':2, 'March':3, 'April':4,
              'May':5, 'June':6, 'July':7, 'August':8,
              'September':9, 'October':10, 'November':11, 'December':12}

    def __repr__(self):
        s = "\nWeather Module Settings:\n\n"
        for key, value in Settings.__dict__.items():
            if type(key) != str:
                continue
            if key.startswith('_'):
                continue
            if type(value)==dict and 'doc' in value:
                s = s + '{:<8} dict: {}\n'.format((key+':'), value['doc'])
            else:
                s = s + '{:<8} {}\n'.format((key+':'), repr(value))
        return s


st = Settings()  # st contains system settings

class WxDF(pd.DataFrame):
    """Weather data management class

    Requires a table of cities (at least one) in the same directory as the
    calling program. The table must be a CSV file named 'Cities.csv' with the
    following headings:
        city, station, path

    Examples
    --------
    ::

        wf = WxDF()       # returns first city in _cf list
        wf = WxDF(3)      # returns city in _cf.iloc[3]
        wf = WxDF(3, df)  # returns WxDF from df, with attributes from _cf
        wf = WxDF(df)     # returns WxDF from df, with attributes empty

    Typing `wf` by itself in iPython will print a summary of the data.
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
    _metadata = ['city','period','type','path','station','id','baseline']

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
                df = pd.read_csv(st.basepath+WxDF._cf.path[id],
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

        # Add a minimum 10-year baseline. Try period up to 1920, otherwise
        # try 1961-1990, otherwise 10 years from beginning, otherwise,
        # just first year.
        # Pre-1920 is the pre-industrial period. 1961-1990 highlights change
        # since around 1975 and is commonly used in literature.
        gap = 10
        if type(self.index[0]) == pd.Timestamp:
            fy = self.index[0].year
            ly = self.index[-1].year
        else:
            fy = self.index[0]
            ly = self.index[-1]
        if fy <= 1920 - gap:
            bls = fy
            ble = 1920
        elif fy <= 1990 - gap:
            bls = max([1961, fy])
            ble = 1990
        elif ly - fy >= gap-1:
            bls = fy
            ble = bls + gap-1
        else:
            bls = ble = fy
        self.baseline = [bls, ble]

        return

    @property
    def _constructor(self):
        return WxDF

    def __str__(self):
        """Return a formatted summary of the data
        """
        hMap = {'Data Quality':'Qual', 'Max Temp (°C)':'Max Temp',
                'Min Temp (°C)':'Min Temp', 'Mean Temp (°C)':'Mean Temp',
                'Total Precip (mm)':'Precip'}

        def GetLine(r):
            """Format a row into a string
            """
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

        Parameters
        ----------
        year : int opt default current year
            Year to retrieve. Defaults to current year.
        raw : boolean opt default False
            If True, do no explicit conversion of supplied data. Use this
            to help debug.
        """
        if year is None:
            year = dt.date.today().year
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

        Parameters
        ----------
        sYear : int opt
            Start year, defaults to year of last data point.
        eYear : int opt
            End year, defaults to current year.
        """
        def Combine(orig, new):
            new.dropna(thresh=5, inplace=True)
            for row in new.index:
                orig.loc[row] = new.loc[row]

        if (eYear is None):
            eYear = dt.date.today().year
        if (sYear is None):
            sYear = self.index[self.GetLastDay()].year
        for theYear in range(sYear, eYear+1):
            nf = self._GetData(theYear)
            Combine(self, nf)

    def Save(self):
        """Save consolidated weather data into a .csv file. Directories are
        created as required.
        """
        file = "".join([st.basepath, self.path])
        p = pathlib.Path(file)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(file,
              float_format="% .1f")

    def GetBaseAvg(self, col=None, range=None):
        """Get the average value over the baseline period for a column.

        Parameters
        ----------
        col : int or str default None
            Column to get average for. Can be the column number, or its name.
            If None, all columns are returned.
        range : list of ints opt default None
            Optional range to compute over if not standard baseline. Must be
            a list with the start and end years (inclusive), eg
            ``range = [1975, 1990]``
        """
        if col and type(col) is int:
            col = self.columns[col]
        if not range:
            bls = self.baseline[0]
            ble = self.baseline[1]
        else:
            bls = range[0]
            ble = range[1]
        if col:
            return self.loc[bls:ble, col].mean()
        else:
            return self.loc[bls:ble].mean()


    def GetFirstDay(self):
        """Return index to first day with valid data.

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

        Parameters
        ----------
        col :   int
            Column to be combined.
        func :  function opt default np.mean
            Function to use for combining. np.min, np.max and np.sum are
            also useful.

        Returns
        -------
        Returns dataframe (not WxDF) with the grouped monthly data in each
        column.

        Note
        ----
        Only works for 1 column at a time due to extra complexity of multi-level
        axes when months are already columns.
        """

        label = self.columns[col]
        avgs = self.pivot_table(values=[label],
                              index=['Year'],
                              columns=['Month'],
                              aggfunc=func)
        avgs = avgs[label]  # turn into simple dataframe for simplicity
        colnames = dict(zip(list(range(13)), st.monthS))
        avgs.rename(columns=colnames, inplace=True)
        mf = WxDF(self.id, avgs)
        mf.period = 'monthly'
        mf.type = func.__name__ + ' ' + label
        return mf

    def GetMonth(self, cols=[4, 6], month=None, func=np.mean):
        """Convert daily data to yearly data for a particular month

        Parameters
        ----------
        cols : list opt default [4, 6] (max, min temps)
            List of columns to be combined
        month : int opt
            Month to combine data for. Defaults to current month
        func : function opt default np.mean
            Function that combines the data. np.min, np.max and np.sum are
            also useful.

        Returns
        -------
        pandas.DataFrame containing monthly data by year
        """
        if month == 0:  # Return full year
            return self.GetYears(cols=cols, func=func)

        if month is None:
            month = dt.date.today().month
        labels = list(self.columns[cols])
        yf = self.loc[lambda df: df.Month == month]
        mf = yf.pivot_table(values=labels,
                              index=['Year'],
                              aggfunc=func)
        mf = WxDF(self.id, mf[labels])

        mf.period = 'annual'
        mf.type = func.__name__.title() + ' for ' + st.monthL[month]
        # Columns possibly in wrong order, so make sure they're ordered
        # as given.
        return mf

    def GetYears(self, cols=[4, 6, 8], func=np.mean):
        """Convert daily data to yearly data.

        cols : list of ints opt default [4, 6, 8] (max, min, average temps)
            Columns to be combined. Defaults to min, max, avg temps
        func : function opt default np.mean
            Function to use for combining. np.min, np.max and np.sum are
            also useful.

        Returns
        -------
        pandas.DataFrame with the data grouped by year
        """
        labels = self.columns[cols]
        yf = self.pivot_table(values=list(labels),
                            index=['Year'],
                            aggfunc=func)

        # Now estimate current year based on rest of previous year
        cyr = self.Year[-1]  # get current year
        pyr = cyr-1          # previous year
        dfc = self.loc[lambda df: df.Year == cyr]
        c = dfc.count()  # days in current year

        dfp = self.loc[lambda df: df.Year == pyr]
        dfp = dfp.iloc[:c[labels[0]]] # make number of days same as current year
        pl = dfp.pivot_table(values=list(labels),
                            index=['Year'],
                            aggfunc=func)
        # adjust current year by last year's change from current day to year end
        for i in range(len(labels)):
            yf.iloc[-1, i] = (yf.iloc[-2, i] - pl.iloc[0, i]) + yf.iloc[-1, i]

        yf = WxDF(self.id, yf[labels])
        yf.period = 'annual'
        yf.type = func.__name__.title() + ' for Year'

        return yf

def GridPlot(df, cols=2, title='', fignum=20):
    """Create a series of plots above each other, sharing x-axis labels.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing data to be plotted in separate columns. Column
        names will be used for labels.
    cols : int (opt) default 2
        Number of columns per figure to use.
    title : String (opt) default blank
        Name to use on each figure. Page numbers are added.
    fignum : int (opt) default 20
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

def TempPlot(df, cols=[8], size=21, trend='wma', pad='linear', follow=1,
             fignum=1):
    """Plot indicated columns of data, including the moving average.

    Parameters
    ----------
    df : WxDF
        DataFrame containing daily data. Can be a pandas.DataFrame with a
        .city attribute added.
    cols : list of ints opt default [8] (Mean Temp)
        Columns to plot.
    size : int opt default 21
        Size of the moving average window. Larger values give smoother
        results.
    trend : str ['wma' | 'lowess' | 'ssa'] default 'wma'
        Which smoothing algorithm to use.
    pad : str ['linear' | 'mirror' | None] default 'linear'
        What kind of padding to use on the trend line
    follow : int default 1
        Determines how closely to follow the data. Only used for
        'lowess' (determines the polynomial to use) and 'ssa' (determines
        how many reconstructed principles to use).
    fignum : in opt default 1
        Figure number to use. Useful if multiple plots are required.

    Note
    ----
    Works best with just one column since all raw data is shown and gets
    messy with multiple columns.
    """

    yf = df.GetYears(cols=cols)
    yf = yf - yf.GetBaseAvg()
    cols = yf.columns

    fig = plt.figure(fignum)
    fig.clear()  # May have been used before
    ax = fig.add_subplot(111)

    for col in cols:
        s = yf[col]
        c = st.colors[col]
        a = sm.Smooth(s, size, trend, pad, follow)
        ax.plot(s, 'o-', alpha=st.da, lw=st.dlw, color=c)
        if col == cols[-1]:
            tlabel = trend.upper() + ' Trend'
        else:
            tlabel = ''
        ax.plot(a, '-', alpha=st.ta, lw=st.tlw,
                 label=tlabel, color=c)
        # fit line to recent data
        at.AddRate(s.loc[1970:])

    # Label chart
    plt.ylabel('Temperature Change From Baseline (°C)')
    #plt.xlabel('Year')
    plt.title("Change in " + df.city + "'s Annual Temperature")

    # Annotate chart
    at.Baseline(df.baseline)
    at.Attribute(source=st.source)
    plt.legend(loc=2)

    at.AddYAxis(ax)
    fig.show()
    return


def TrendPlot(df, cols=[4, 6, 8], size=21, change=True, rate=False,
              trend='wma', pad='linear', follow=1, fignum=2):
    """Simple smoothed plots with optional baseline.

    Parameters
    ----------
    df : WxDF
        DataFrame containing daily data. Can be a pandas.DataFrame with a
        .city attribute added.
    cols : list of ints opt default [4, 6, 8] (Max, Min, Avg Temp)
        Columns to plot.
    size : int opt default 21
        Size of the moving average window. Larger values give smoother
        results.
    change : boolean opt default True
        Show change from the baseline
    trend : str ['wma' | 'lowess' | 'ssa'] default 'wma'
        Which smoothing algorithm to use.
    pad : str ['linear' | 'mirror' | None] default 'linear'
        What kind of padding to use on the trend line
    follow : int default 1
        Determines how closely to follow the data. Only used for
        'lowess' (determines the polynomial to use) and 'ssa' (determines
        how many reconstructed principles to use).
    fignum : in opt default 2
        Figure number to use. Useful if multiple plots are required.
    """
    yf = df.GetYears(cols=cols)
    if change:
        yf = yf - yf.GetBaseAvg()
    ma = [sm.Smooth(yf[col], size, trend, pad, follow) for col in yf.columns]
    fig = plt.figure(fignum)
    fig.clear()
    ax = fig.add_subplot(111)
    for i, y in enumerate(ma):
        ax.plot(y, '-', alpha=st.ta, linewidth=st.tlw, label=yf.columns[i])
        if rate: at.AddRate(y.loc[1970:])

    # Annotate
    plt.ylabel('Temperature Change from Baseline (°C)')
    plt.xlabel('Year')
    plt.title("Change in " + df.city + "'s Annual Temperature")
    plt.legend(loc='upper left')
    at.Baseline(df.baseline)
    at.Attribute(source=st.source)

    at.AddYAxis(ax)
    fig.show()
    return

def ErrorPlot(df, cols=[8], size=21, trend='wma', pad='linear', follow=1,
              fignum=3):
    """Show standard deviation of temperature from trend.

    Parameters
    ----------
    df : WxDF
        DataFrame containing daily data. Can be a pandas.DataFrame with a
        .city attribute added.
    cols : list of ints opt default [8] (Mean Temp)
        Columns to plot.
    size : int opt default 21
        Size of the moving average window. Larger values give smoother
        results.
    trend : str ['wma' | 'lowess' | 'ssa'] default 'wma'
        Which smoothing algorithm to use.
    pad : str ['linear' | 'mirror' | None] default 'linear'
        What kind of padding to use on the trend line
    follow : int default 1
        Determines how closely to follow the data. Only used for
        'lowess' (determines the polynomial to use) and 'ssa' (determines
        how many reconstructed principles to use).
    fignum : in opt default 3
        Figure number to use. Useful if multiple plots are required.
    """
    c = st.colors[cols[0]]
    yf = df.GetYears(cols)
    yf = yf - yf.GetBaseAvg()
    col = yf.columns[0]
    ma = sm.Smooth(yf[col], size, trend, pad, follow)
    err = (ma - yf[col])**2
    std = err.mean()**0.5
    fig = plt.figure(fignum)
    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(yf[col], 'o-', lw=st.dlw, alpha=st.da,
            label=' '.join([df.city, col]), color=c)
    #c = line[0].get_color()
    ax.plot(ma, '-', alpha=st.ta, lw=st.tlw,
            label='Trend', color=c)
    ax.fill_between(ma.index, ma.values+std, ma.values-std,
                     color=c, alpha=st.sa, label='68%')
    ax.fill_between(ma.index, ma.values+2*std, ma.values-2*std,
                     color=c, alpha=st.ma, label='95%')

    # Annotate chart
    plt.legend(loc='upper left')
    plt.title("Change in " + df.city + "'s Annual Temperature")
    plt.ylabel('Temperature Change from Baseline (°C)')
    at.Baseline(yf.baseline)
    at.Attribute(source=st.source)

    plt.show()


def RecordsPlot(df, use=[0,1,2,3,4,5], stack=False, fignum=4):
    """Plot all records in daily data.

    Parameters
    ----------
    df : WxDF
        object containing daily data for a location. Can use a
        pandas.DataFrame if df.city comtains the name of the city.
    use : list of int opt default [0,1,2,3,4,5]
        List of records to show. They are any of

        0. Max Day,
        1. Min Day,
        2. Max Night
        3. Min Night
        4. Rain
        5. Snow
    stack : boolean default False
        Show the record counts for each year in a separate stackplot.
    fignum : (opt) default 4
        Figure to use. Useful to keep multiple plots separated.
    """

    def ToNow(t):
        """Take a timestamp and return same day in 2016"""
        return pd.Timestamp(dt.date(2016, t.month, t.day))

    start = 1960  # start date for x-axis
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
    fig = plt.figure(fignum)
    fig.clear()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.1)

    # Set up axis formatting
    monthly = mdates.MonthLocator()
    monthFmt = mdates.DateFormatter('%b')
    blankFmt = mdates.DateFormatter(' ')
    bimonthly = mdates.DayLocator(15)
    ax.set_ylim((dt.date(2015,12,16), dt.date(2017,1,14)))
    ax.yaxis.set_major_locator(monthly)
    ax.yaxis.set_minor_locator(bimonthly)
    ax.yaxis.set_minor_formatter(monthFmt)
    ax.yaxis.set_major_formatter(blankFmt)
    # ticks must be set before the first plot, or they will be locked in
    ax.set_xticks(np.arange(start, 2021, 5))
    ax.set_xlim((start-2, 2022))

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
        # drop any dates before 1960
        for i in range(len(x)):
            if x[i] >= start:
                break
        x = x[i:]
        y = y[i:]
        ax.plot(x, y,
                 p[2],
                 label=p[0], zorder=p[3])

    # Plot records data
    ax.legend(loc='upper left', ncol=6,
              bbox_to_anchor=(0, -0.04), handlelength=0.8)
    plt.title('Daily Weather Records for ' + df.city)

    # Add second y-axis
    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks())
    ax2.set_ylim(ax.get_ylim())
    ax2.grid(False)
    ax2.yaxis.set_major_locator(monthly)
    ax2.yaxis.set_minor_locator(bimonthly)
    ax2.yaxis.set_minor_formatter(monthFmt)
    ax2.yaxis.set_major_formatter(blankFmt)
    ax.tick_params(axis='y', which='minor', color=(0,0,0,0))
    ax2.tick_params(axis='y', which='minor', color=(0,0,0,0))

    plt.show()


    if not stack: return

    # Plot number of records per year in a stacked bar chart
    x = list(counts.index)
    y = list(range(len(counts.columns)))
    for i, v in enumerate(counts.columns):
        y[i] = list(counts[v])
    fig = plt.figure(fignum+1, figsize=(17, 9))
    fig.clear()
    plt.axis([start, 2020, 0, 50])
    plt.stackplot(x, y)


    plt.legend(counts.columns)
    plt.title('Records per Year for ' + df.city)

    plt.show()
    print('Done')
    return


def PrecipPlot(df, fignum=5):
    """Go through all data and plot every day where it rained or snowed.

    Parameters
    ----------
    df : WxDF
        object containing daily data for a location. Can use a
        pandas.DataFrame if df.city comtains the name of the city.
    fignum : int opt default 5
        Figure to use. Useful to keep multiple plots separated.
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

    Parameters
    ----------
     df : WxDF
        object containing daily data for a location. Can use a
        pandas.DataFrame if df.city comtains the name of the city.
    fignum : int opt default 6
        Figure to use. Useful to keep multiple plots separated.
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
                 linewidth=st.tlw,
                 alpha=st.ta,
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

    Parameters
    ----------
    df : WxDF
        object containing daily data for a location. Can use a
        pandas.DataFrame if df.city comtains the name of the city.
    fignum : int opt default 7
        Figure to use. Useful to keep multiple plots separated.
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

def MonthRangePlot(nf, month=None, combine=True,
                   trend='wma', pad='linear', follow=1, fignum=8):
    """Get expected high and low temperatures for the supplied month.

    Parameters
    ----------
    nf : pandas.DataFrame
        daily data contained in a pandas DataFrame
    month : int opt default None
        Desired month. The default gives current month. ``month=0`` gives
        plot for entire year.
    combine : boolean opt default True
        Combine the maximum and minimum temperatures onto one plot. Otherwise
        use two separate plots (which is easier to read).
    trend : str ['wma' | 'lowess' | 'ssa'] default 'wma'
        Which smoothing algorithm to use.
    pad : str ['linear' | 'mirror' | None] default 'linear'
        What kind of padding to use on the trend line
    follow : int default 1
        Determines how closely to follow the data. Only used for
        'lowess' (determines the polynomial to use) and 'ssa' (determines
        how many reconstructed principles to use).
    fignum : int opt default 8
        Figure number to use. Override if you want to see more than one plot
        at a time.
    Note
    ----
    Uses moving average to calculate the mean temperatures, and the standard
    deviation from this.
    """
    short = 5  # length of short weighting window
    long = 19  # length of long weighting window
    vlong = 61 # length of daily weighting window

    if month is None:
        month = dt.date.today().month
    maxc = nf.columns[4] # max:0/4, min:1/6, avg:2/8
    minc = nf.columns[6]
    avgc = nf.columns[8]
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
        af[c] = sm.Smooth(df[c], size=vlong, trend='wma', pad=None)
    # Get the error and standard deviation
    for c in [maxc, minc]:
        ef[c] = df[c] - af[c]
        sf[c] = sm.Smooth(ef[c]**2,
                      size=vlong, trend='wma', pad=None)**0.5

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
        mx[c] = sm.Smooth(mx[c], size=short,
                          trend=trend, pad=pad, follow=follow)
        mn[c] = sm.Smooth(mn[c], size=short,
                          trend=trend, pad=pad, follow=follow)
        sf[c] = sm.Smooth(sf[c], size=long,
                          trend=trend, pad=pad, follow=follow)
    for c in [maxc, minc, avgc]:
        af[c] = sm.Smooth(af[c], size=long,
                          trend=trend, pad=pad, follow=follow)

    umaxt = af[maxc] + sf[maxc]
    lmaxt = af[maxc] - sf[maxc]
    umint = af[minc] + sf[minc]
    lmint = af[minc] - sf[minc]
    index = af.index

    # Get unsmoothed values for last year
    if month != 0:
        avm = nf.GetMonth([4, 6, 8], month, func=np.mean)
        mxm = nf.GetMonth([4, 6, 8], month, func=np.max)
        mnm = nf.GetMonth([4, 6, 8], month, func=np.min)
    else:
        avm = nf.GetYears([4, 6, 8], func=np.mean)
        mxm = nf.GetYears([4, 6, 8], func=np.max)
        mnm = nf.GetYears([4, 6, 8], func=np.min)

    # PLOTTING
    title = 'Temperature Range in '+df.city+' for '+ st.monthL[month]
    fig = plt.figure(fignum)
    fig.clear()
    if not combine:  # create two separate plots
        plt.subplots_adjust(hspace=0.001, wspace=0.1,
                            left=0.08, right=0.92,
                            bottom=0.05, top=0.95)
        ax0 = plt.subplot(2, 1, 1)
        ax1 = plt.subplot(2, 1, 2, sharex=ax0)
        xt = ax0.get_xticklabels()
        plt.setp(xt, visible=False)
        fig.suptitle(title)
    else:  # just put everything on one plot
        ax0 = plt.subplot(1, 1, 1)
        ax1 = ax0
        plt.title(title)

    ax0.fill_between(index, mxm[maxc], mnm[maxc],
                     color='C0', alpha=st.ma, label='Upper/Lower Highs')
    ax1.fill_between(index, mxm[minc], mnm[minc],
                     color='C1', alpha=st.ma, label='Upper/Lower Lows')
    ax0.fill_between(index, umaxt, lmaxt,
                     color='C0', alpha=st.sa, label='68% Range Highs')
    ax1.fill_between(index, umint, lmint,
                     color='C1', alpha=st.sa, label='68% Range Lows')
    ax0.plot(af[maxc], 'C0-', lw=2, alpha=st.ta, label='Average Highs')
    ax1.plot(af[minc], 'C1-', lw=2, alpha=st.ta, label='Average Lows')
    if combine:
        ax0.plot(af[avgc], 'C2-', lw=st.tlw, alpha=st.ta, label='Average Daily')

    # Add current available month as distinct points
    ly = avm.index[-1]
    marks = ['^', 'o', 'v']
    maxvals = [mxm.iloc[-1,0], avm.iloc[-1,0], mnm.iloc[-1,0]]
    minvals = [mxm.iloc[-1,1], avm.iloc[-1,1], mnm.iloc[-1,1]]
    maxt = [' Max\n Day', ' Avg\n Day', ' Min\n Day']
    mint = [' Max\n Ngt', ' Avg\n Ngt', ' Min\n Ngt']
    maxt[0] = str(ly) + '\n' + maxt[0]
    for mk, mxv, mnv, mxt, mnt in zip(marks, maxvals, minvals, maxt, mint):
        ax0.plot(ly, mxv, color='C0', marker=mk)
        ax1.plot(ly, mnv, color='C1', marker=mk)
        ax0.text(ly, mxv, mxt, ha='left', va='center', size='small')
        ax1.text(ly, mnv, mnt, ha='left', va='center', size='small')

    # Annotate
    ax0.set_ylabel('Temperature °C')
    if not combine:
        ax1.set_ylabel('Temperature °C')

    def mid(x, f=.5):
        """Get the midpoint between min and max, or a fraction if supplied
        """
        mn = min(x)
        mx = max(x)
        return (mx-mn)*f + mn

    txt0 = ['Hottest Day', 'Coldest Day',
            'Average High', '68% Day Spread']
    txt1 = ['Hottest Night', 'Coldest Night',
            'Average Low', '68% Night Spread']
    va0 = ['top', 'bottom', 'bottom', 'top']
    va1 = ['top', 'bottom', 'top', 'bottom']
    yrs = len(index)
    xx0 = [yrs*.1, yrs*.1, yrs*.35, yrs*.2]
    xx1 = [yrs*.1, yrs*.1, yrs*.35, yrs*.2]
    xx0 = [int(x) for x in xx0]
    xx1 = [int(x) for x in xx1]
    yy0 = [mid(mxm[maxc],.8), mid(mnm[maxc],.2),
           af[maxc].iloc[xx0[2]], umaxt.iloc[xx0[3]]]
    yy1 = [mid(mxm[minc]), mid(mnm[minc],.2),
           af[minc].iloc[xx1[2]], lmint.iloc[xx1[3]]]
    for t, v, x, y in zip(txt0, va0, xx0, yy0):
        ax0.text(index[x], y, t, va=v,
                 ha='center', color='darkred', size='smaller')
    for t, v, x, y in zip(txt1, va1, xx1, yy1):
        ax1.text(index[x], y, t, va=v,
                 ha='center', color='darkblue', size='smaller')
    if combine:
        x = index[xx0[2]]
        y = af[avgc].iloc[xx0[2]]
        ax0.text(x, y, 'Month Average',
                 ha='center', va='bottom', size='smaller')

    at.Attribute(source=st.source)

    at.AddYAxis(ax0)
    if not combine: at.AddYAxis(ax1)
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
    at.Baseline(df.baseline)

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
    at.Baseline(df.baseline)
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
