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
   * Plot - Temperature plot with optional annotations
   * RecordsPlot - Show all records on one graph
   * RecordsRatioPlot - Ratio of high to low records by decade
   * DayPlot - Matrix of every day colored by the type of day
   * DayCountPlot - Shows how types of days change over time
   * DayThreshPlot - How many days per year above or below threshold
   * TemperatureCountPlot - Days in each temperature range per year
   * WarmPlot - Shows how the warm season changes over time
   * WarmDaysPlot - How many days in the warm season
   * SnowPlot - Plots first and last snowfall
   * TopPrecipPlot - Amount of precipitation on top days
   * StormPlot - Amount of precipitation of top storms (over multiple days)
   * MonthRange - High and Low range for a single month

Miscellaneous
-------------
   * GridPlot - Put several plots on one page
   * CompareSmoothing - Show how Lowess and WMA compare for trends
   * CompareWeighting - Show how different weight windows compare

Requirements
------------
   Requires Python 3 (tested on 3.6.1, Anaconda distribution)
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import Smoothing as sm
import Annotate as at
import pathlib
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib.patheffects as path_effects


#%precision 2
pd.options.display.float_format = '{:.1f}'.format  # change print format
plt.style.use('weather')
# http://matplotlib.org/users/style_sheets.html
# matplotlib.style.available shows available styles
# matplotlib.style.library is a dictionary of available styles
# user styles can be placed in ~/.matplotlib/

#Default settings for trend lines
trendDefault = {'size':21, 'trend':'wma', 'pad':'linear', 'follow':1}
""" size : int default 21
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
"""

class Settings():
    """Simple class to hold settings, making out of scope variables more
       obvious. Use str(obj) to get list of settings, or just 'obj' at the
       command line.
    """
    _desc = {}  # description of variables
    basepath = '/Users/Dan/Documents/Weather/Stations/'
    source = "Data: Environment Canada"
    tlw = 4; _desc['tlw']='trend linewidth'
    dlw = 1; _desc['dlw']='data linewidth'
    ta = 0.99; _desc['ta']='trend alpha'
    da = 0.3; _desc['da']='data alpha'
    sa = 0.15; _desc['sa']='std dev alpha'
    ma = 0.1; _desc['ma']='max/min alpha'
    spring = [3, 4, 5]; _desc['spring']='months for spring'
    summer = [6, 7, 8]; _desc['summer']='months for summer'
    fall = [9, 10, 11]; _desc['fall']='months for fall'
    winter = [12, 1, 2]; _desc['winter']='months for winter'


    colors = {'doc':'Color from cycle to use per column',
              'Max Temp (°C)':'C0', 4:'C0',
              'Min Temp (°C)':'C1', 6:'C1',
              'Mean Temp (°C)':'C2', 8:'C2',
              'Total Rain (mm)':'C2', 14:'C2',
              'Total Snow (cm)':'C1', 16:'C1',
              'Total Precip (mm)':'C3', 18:'C3'} # colors to use per column
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
        # Return list of variables and their descriptions in printable format
        s = "\nWeather Module Settings:\n\n"
        for key, value in Settings.__dict__.items():
            if type(key) != str:
                continue
            if key.startswith('_'):
                continue
            if type(value)==dict and 'doc' in value:
                s += '{:<8} dict: {}\n'.format((key+':'), value['doc'])
            else:
                s += '{:<8} {:<12}{}\n'.format((key+':'),
                                              repr(value),
                                              self._desc.get(key,''))
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
    # Specify column locations here. These should be used by routines
    # The column names can be determined by the df.columns list
    temps = [4, 6, 8]  # max, min, average temps
    precips = [14, 16, 18]  # rain, snow, precipitation
    (tmx, tmn, tav) = temps
    (rn, sn, pr) = precips
    wind = 24
    wdir = 22
    qual = 3  # data quality column

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
        labels = ['Qual', 'Tmax', 'Tmin', 'Tavg', 'Rain', 'Snow', 'Prec']
        names = ['Data Quality', 'Max Temp (°C)', 'Min Temp (°C)',
               'Mean Temp (°C)', 'Total Rain (mm)', 'Total Snow (cm)',
               'Total Precip (mm)']
        cols = [self.qual, self.tmx, self.tmn,
                self.tav, self.rn, self.sn, self.pr]
        # make map of column name to short label
        hMap = dict(zip(names, labels))
        mincw = 7  # minimum column width

        def GetLine(r):
            """Format row r into a string
            """
            # hdgs and lbl are defined outside function
            if hasattr(self.index[0], 'year'):
                st = str(self.index[r].date()).ljust(10)
            else:
                st = str(self.index[r]).ljust(10)
            for i, c in enumerate(lbl):
                num = max(mincw, len(hdgs[i]))
                if type(self[c].iloc[r]) is np.float64:
                    st = st + '{:.1f}'.format(self[c].iloc[r]).rjust(num)
                else:
                    st = st + str(self[c].iloc[r]).rjust(num)
            return st

        # Set up the headings to use. The period attribute will determine
        # what to use. If period does not exist, just show first 4 columns.
        hdgs = '    Undefined'
        if not hasattr(self, 'period'):
            num = min(4, len(self.columns))
            lbl = list(self.columns[0:num])
            hdgs = [l.rjust(max(mincw,len(l))) for l in lbl]
        elif self.period == 'daily':
            full = list(hMap.keys())
            avail = list(self.columns)
            lbl = []
            for h in full:
                if h in avail:
                    lbl.append(h)
            hdgs = [hMap[h] for h in lbl]
            hdgs = [h.rjust(max(mincw,len(h))) for h in hdgs]
        elif self.period == 'monthly':
            cols = [0, 4, 8, 11] # months to show
            lbl = list(self.columns[cols])
            hdgs = [l.rjust(max(mincw,len(l))) for l in lbl]
        elif self.period == 'annual':
            lbl = list(self.columns)
            hdgs = [l.rjust(max(mincw,len(l))) for l in lbl]
        first = self.GetFirstDay()
        last = self.GetLastDay()
        # create heading
        s = ''
        if hasattr(self, 'city') and self.city is not None:
            s = "City: {0}  Type: {1}\n".format(self.city, self.type)
        s = s + "Date      " + "".join(hdgs)
        # add beginning lines
        if first > 0:
            s = '\n'.join([s, GetLine(0)])
        if first > 2:
            s = '\n'.join([s, '...'])
        if first > 1:
            s = '\n'.join([s, GetLine(first-1)])
        if (last-first)<32:
            for i in range(first, last+1):
                s = '\n'.join([s, GetLine(i)])
        else:
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

        # Drop last year if incomplete
        date = df.index[df.GetLastDay()]
        if date.dayofyear < 365:
            yf.drop(date.year, inplace=True)

        yf = WxDF(self.id, yf[labels])
        yf.period = 'annual'
        yf.type = func.__name__.title() + ' for Year'

        return yf

class Texture(object):
    """Class for using agg_filter to apply texture to matplotlib patches.
        Basic information comes from `Demo Agg Filter
        <https://matplotlib.org/gallery/misc/demo_agg_filter.html>`_

        Parameters
        ----------
        darkness : float (0 - 1) default 0.15
            How much darker to make the patch color. The patch colors are
            multiplied by (1 - darkness) at random locations. Levels
            at 1/3 and 2/3 amounts are also used.
        prob : float (0 - 1) default 0.3
            Probability that any point will be made darker

        Usage
        -----
        Make a filter object with the desired parameters, eg::

            filt = Texture(darkness=.1, prob=.08)

        then use the filter when drawing something you want textured::

            ax.fill_between([0,1], [1,1], agg_filter=filt)

        You can have multiple texture styles by creating multiple filter
        objects::

            light_texture = Texture(darkness=.05, prob=.05)
            dark_texture = Texture(darkness=.15, prob=.2)
            ax.fill_between([0,1], [1,1], agg_filter=light_texture)
            ax.fill_between([1,2], [1,1], agg_filter=dark_texture)

    """
    def __init__(self, darkness=.15, prob=.3):
        """Initialize filter object with the persistent data required.
        """
        # Create list of values to choose from. '1.' means original
        # image value will be unchanged. '.85' will cause image to be
        # darkened to 85% of original value.
        choice = list(1. - np.array([1,2,3]) * darkness/3)
        choice.append(1.)
        # [.95, .90, .85, 1.]
        self.choice = choice
        self.p = 3 * [prob/3] + [1. - prob]  # probabilities
        # [.1, .1, .1, .7]

    def __call__(self, src, dpi):
        """Appy texture to supplied image"""
        sh = src.shape
        shape = (sh[0], sh[1], 1)
        noise = np.random.choice(self.choice, shape, p=self.p)
        res  = src * noise
        # Top layer likely has alpha with patch shape, so copy from original
        res[:, :, 3] = src[:, :, 3]
        return res, 0, 0  # 0, 0 are likely x,y offsets for, eg, dropshadows


def _AddEOY(df, col, offset=0, ax=None, legend=True, onlymean=True,
            func=np.mean):
    """Make an estimate of the mean temperature for the last year in data.

    Parameters
    ----------
    df : WxDF or pd.DataFrame
        Data to be analyzed. Expects columns to be in WxDF format.
    col : int
        Column to use for calculation
    offset : int default 0
        Offset to use, eg for when plotting against a baseline. This is
        subtracted from the actual value.
    ax : matplotlib.Axis or None, default None
        Axis to plot on. Will not plot if None. If provided, will plot a box
        plot showing the mean, 2-sigma (95%), min and max values.
    legend : bool default True
        Flag to include labels in legend.
    onlymean : bool default True
        Only add mean to graph
    func : function default np.mean
        function to use for aggregating annual data. Use np.mean for
        temperatures, and np.sum for precipitation.

    Returns
    -------
    mean : float
        Estimated final mean temperature for final year
    sigma : float
        Standard deviation from all previous years for the rest of the year.
    max : float
        Maximum seen deviation from temperature to present day of year.
    min : float
        Minimum seen deviation from temperature to present day of year.

    Note
    ----
    Return values have the offset subtracted from them.
    """
    if type(col) == str:
        tcol = col
    else:
        tcol = df.columns[col]
    tcols = df.columns[[0,4,6,8,14,16,18]]
    date = df.index[df.GetLastDay()]
    yr = date.year
    dy = date.dayofyear
    # get days in full year
    fy = pd.Timestamp(dt.date(yr, 12, 31)).dayofyear
    df = df.loc[:,tcols] # only use useful columns
    df['dy'] = df.index.dayofyear

    # For all previous years, get days up to and including last day,
    # and days afterwards. Then get the sum for each year.
    bf = df.loc[df.dy <= dy] # beginning
    ef = df.loc[df.dy > dy]  # end
    yf = bf.groupby('Year').aggregate(func)
    yf['end'] = ef[['Year',tcol]].groupby('Year').aggregate(func)

    # The difference between beginning of year average temperature should be
    # correlated with the end of year temp, so calculate this for every year,
    # then get the stats for the differences to determine how the end of the
    # last year should end up.
    yf['diff'] = yf['end'] - yf[tcol]
    # Get results weighted by amount of year left
    bw = dy/fy  # beginning weight
    ew = 1.0 - bw  # end weight
    yb = yf.loc[yr, tcol]  # beginning temp
    yAvg = yb * bw + (yb + yf['diff'].mean()) * ew - offset
    yMax = yb * bw + (yb + yf['diff'].max()) * ew - offset
    yMin = yb * bw + (yb + yf['diff'].min()) * ew - offset
    yStd = yf['diff'].std(ddof=0) * ew

    if ax:
        ys = str(yr)
        ps = ms = es = ''
        if legend:
            ps = ys+' (est)'
            ms = ys+' Min/Max'
            es = ys+' 95% range'
        ax.plot(yr, yAvg, 'ko', label=ps)
        if not onlymean:
            ax.plot([yr, yr], [yMax, yMin], 'k_-', lw=1., label=ms, ms=7,
                    alpha=0.8)
            ax.plot([yr, yr], [yAvg+2*yStd, yAvg-2*yStd], '-', color='orange',
                    alpha=0.5, lw=7, label=es)

    return yAvg, yStd, yMax, yMin

def _GetTrendArgs(sTrend=trendDefault, **kwargs):
    """Utility to return a new dict with updated trend keyword values

    Parameters
    ----------
    sTrend : dict default trendDefault
        starting dictionary of trend keywords to be updated
    **kwargs : keywords and values with which to update the trend dict
        Must supply '**kwargs', not 'kwargs', or else kwargs will replace
        sTrend.

    Returns
    -------
    newTrend : updated dictionary of trend keywords and values.
    """
    newTrend = sTrend.copy()
    newTrend.update(kwargs)
    return newTrend

def Plot(df, rawcols=None, trendcols=None, ratecols=None,
             func=None, change=True, est=True, **kwargs):
    """Plot indicated columns of data, including the moving average.

    Parameters
    ----------
    df : WxDF
        DataFrame containing daily data. Can be a pandas.DataFrame with a
        .city attribute added.
    rawcols : list of ints default [8] (Mean Temp)
        Columns to plot as raw data.
    trendcols : list of ints default [8] (Mean Temp)
        Columns to plot as trend lines.
    ratecols : list of ints default [8] (Mean Temp)
        Columns to add rate lines to.
    func : function default np.mean
        Function used to aggregate the annual data. Use np.sum
        for precipitation.
    change : bool default True
        Flag determines if change from baseline desired.
    est : bool default True
        Include current incomplete year as an estimate.
    kwargs : **dict
        keywords to pass to the smoothing function that override the default
        values. See Smoothing.Smooth or the trendDefault definition for
        explanation.

    Notes
    -----
    If only rawcols are provided, trend and rate lines will be added. Use
    empty lists if these aren't desired.

    It is possible to use both temperature and precipitation columns, but
    this will generally be not useful, and the labels will be incorrect.
    """

    trend = _GetTrendArgs(**kwargs)

    # get a list of all desired columns (check for None)
    allcols = set()
    for s in [rawcols, trendcols, ratecols]:
        if s: allcols = allcols.union(set(s))
    allcols = list(allcols)

    # set up defaults. If first supplied columns is temperature, make
    # defaults temperature, otherwise precipitation.
    if len(allcols)==0 or (allcols[0] in df.temps):
        unitstr = ' (°C)'
        typestr = 'Temperature'
        ratestr = '{:.2f}°C/decade'
        if not func: func = np.mean
        if not rawcols: rawcols = [df.tav]
        if not trendcols: trendcols = rawcols
        if not ratecols: ratecols = trendcols
    else:
        unitstr = ' (mm/cm)'
        typestr = 'Precipitation'
        ratestr = '{:.1f}/decade'
        if not func: func = np.sum
        if not rawcols: rawcols = [df.pr]  # total precipitation
        if not trendcols: trendcols = rawcols
        if not ratecols: ratecols = trendcols
    allcols = list(set().union(set(rawcols), set(trendcols), set(ratecols)))

    yf = df.GetYears(cols=allcols, func=func)
    offset = yf.GetBaseAvg()  # offset is used later
    if change:
        yf = yf - offset
        ychstr = ' Change From Baseline'
        chstr = 'Change in '
    else:
        offset[:] = 0
        ychstr = ''
        chstr = ''
    cols = yf.columns

    fig = plt.figure(df.city+'_Plot')
    fig.clear()  # May have been used before
    ax = fig.add_subplot(111)

    # Create legend entries manually
    handles = []
    if len(rawcols) > 0:
        line = mlines.Line2D([], [], color='k', marker='o',
                             alpha=st.da, lw=st.dlw, label='Raw Data')
        handles.append(line)
    if len(trendcols) > 0:
        line = mlines.Line2D([], [], color='k', alpha=st.ta, lw=st.tlw,
                             label=trend['trend'].upper()+' Trend')
        handles.append(line)

    for col in cols:
        s = yf[col]
        if est:
            r = _AddEOY(df, col, offset[col], func=func)
            s[s.index[-1]+1] = r[0]  # add current year estimate
        c = st.colors[col]

        # add a legend entry for this color
        line = mpatches.Patch(color=c, label=col)
        handles.append(line)

        a = sm.Smooth(s, trend)
        if col in df.columns[rawcols]:
            ax.plot(s, 'o-', alpha=st.da, lw=st.dlw, color=c)
        if col in df.columns[trendcols]:
            ax.plot(a, '-', alpha=st.ta, lw=st.tlw, color=c)
        if col in df.columns[ratecols]:
            # fit line to recent data
            # Use smoothed line for rate since different methods may reduce
            # influence of outliers.
            at.AddRate(a.loc[1970:], label=ratestr)

    # Label chart
    plt.ylabel(typestr + ychstr + unitstr)
    plt.title(chstr + df.city + "'s Annual " + typestr)

    # Annotate chart
    if change:
        at.Baseline(df.baseline)
    at.Attribute(source=st.source, ha='left')
    plt.legend(handles=handles, loc=2)

    at.AddYAxis(ax)
    fig.show()
    return

def RecordsPlot(df, use=[0,1,2,3,4,5], stack=False):
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
    """

    start = 1960  # start date for x-axis
    # set up data for each set of records:
    # [Name, df column, mark color and format, zorder]
    props = [
             ['Max Day', df.tmx, '^', 'C0', 6, float.__gt__, -100.0],
             ['Min Day', df.tmx, 'v', 'C3', 5, float.__lt__, 100.0],
             ['Max Night', df.tmn, '^', 'C5', 4, float.__gt__, -100.0],
             ['Min Night', df.tmn, 'v', 'C1', 3, float.__lt__, 100.0],
             ['Rain', df.rn, 'o', 'C2', 2, float.__gt__, -100.0],
             ['Snow', df.sn, 'H', 'C5', 1, float.__gt__, -100.0],
             ]
    props = [props[i] for i in use]
    columns = [p[0] for p in props]
    fig = plt.figure(df.city+'_Records')
    fig.clear()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.1)

    # Set up axis formatting
    at.MonthFmt(ax)
    # ticks must be set before the first plot, or they will be locked in
    ax.set_ylim((dt.date(2015,12,16), dt.date(2017,1,14)))
    ax.set_xticks(np.arange(start, 2021, 5))
    ax.set_xlim((start-2, 2022))

    # will look up records by month/day (1-indexed)
    r = np.zeros((13, 32), dtype=float)
    yrmin = df.index[0].year
    yrmax = df.index[-1].year
    counts = np.zeros((len(use), (yrmax-yrmin+1)), dtype=int)
    for pi, p in enumerate(props):
        print(p[0])
        # contains running list of records. x is the year, y is the date.
        x = []
        y = []
        # choose appropriate comparison function. The 'Min' records are '<'
        compare = p[5]
        r[:,:] = p[6]
        for i in range(len(df.index)):
            s = df.iat[i, p[1]]
            month = df.iat[i, 1]
            day = df.iat[i, 2]
            if compare(s, r[month,day]):
                r[month,day] = s
                x.append(df.index[i].year)
                y.append(df.index[i].replace(year=2016))
                counts[pi, df.iat[i,0]-yrmin] += 1
        # drop any dates before 1960
        for i in range(len(x)):
            if x[i] >= start:
                break
        x = x[i:]
        y = y[i:]
        ax.plot(x, y,
                 marker=p[2], color=p[3], ls='',
                 label=p[0], zorder=p[4])

    # annotate axes
    ax.legend(loc='upper left', ncol=6,
              bbox_to_anchor=(0, -0.04), handlelength=0.8)
    plt.title('Daily Weather Records for ' + df.city)

    # Add second y-axis
    at.AddYAxis(ax, month=True)

    plt.show()


    if not stack: return

    # Plot number of records per year in a stacked bar chart
    x = list(range(yrmin, yrmax+1))
    fig = plt.figure(df.city+'_Records_Bar')
    fig.clear()
    ax = fig.add_subplot(111)
    plt.axis([start, 2020, 0, 45])
    ax.set_xticks(np.arange(start, 2021, 5))
    ax.set_yticks(np.arange(0, 46, 5))

    plt.stackplot(x, counts, alpha=0.7)

    plt.legend(columns)
    plt.title('Records per Year for ' + df.city)

    plt.show()
    print('Done')
    return

def RecordsRatioPlot(df):
    """Find ratio of warm records to cold records for each year
    """

    grp = 10  # number of years to group together
    yr = 'Year'
    # create dataframe to hold yearly count of records
    cf = pd.DataFrame(index=np.arange(df.index[0].year,
                                      df.index[-1].year+1, grp))
    # will look up records by day of year (1-indexed)
    cr = np.zeros(367, dtype=float)

    for col, ct in zip(df.temps[:2], ['D', 'N']):
        for t, comp, lim in zip(['H', 'L'],
                         [float.__gt__, float.__lt__],
                         [-1000, 1000]):
            r = []  # list of record days
            cr[:] = lim
            print(ct+t)
            for i in range(len(df.index)):
                d = df.index[i]
                val = df.iat[i, col]
                if comp(val, cr[d.dayofyear]):
                    cr[d.dayofyear] = val
                    # add date, grouped year, and value to list
                    r.append([d, np.floor(d.year/grp)*grp, val])
            cn = df.columns[col]
            rf = pd.DataFrame(r, columns=['Date', yr, cn])
            cf[ct+t] = rf[[yr, cn]].groupby(yr).count()
    cf['H'] = cf['DH'].add(cf['NH'], fill_value=0)
    cf['L'] = cf['DL'].add(cf['NL'], fill_value=0)
    cf['H/L'] = cf['H']/cf['L']
    cf['DH/NL'] = cf['DH']/cf['NL']

    fig = plt.figure(df.city+'_Records_Ratio')
    fig.clear()
    ax = fig.add_subplot(111)
    xticks = list(cf.index)
    xlabels = ["{}s".format(s) for s in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.plot(cf['H/L'], 'o-', color='C0', label='',
            lw=st.tlw, alpha=st.ta)
    ax.set_title('Ratio of Record Highs to Lows (Day + Night)\n'
                 'for ' + df.city + ', Grouped by Decade')
    ax.set_ylabel('Ratio of Record Highs to Lows')
    ax.axhline(1, ls='--', color='k', lw=1)
    at.Attribute(ax, ha='left', va='top', source=st.source)

    at.AddYAxis(ax)
    plt.show()

    return cf

def DayPlot(df, start=1940, use = [0,1,2,3,4,5,6,7]):
    """Go through all data and plot what the weather was like for each day.

    Parameters
    ----------
    df : WxDF
        object containing daily data for a location. Can use a
        pandas.DataFrame if df.city comtains the name of the city.
    start : int default 1940
        Year to start the plot.
    use : list of int default [0,1,2,3,4,5,6]
        Data to plot.

    Note
    ----
    Colours are determined by the stylesheet
    """

    fig = plt.figure(df.city+'_Days')
    fig.clear()
    ax = fig.add_subplot(111)
    # Set up axis formatting
    at.MonthFmt(ax)
    # ticks must be set before the first plot, or they will be locked in
    # Adds space, and assumes data is for 2016.
    ax.set_ylim((dt.date(2015,12,16), dt.date(2017,1,14)))
    if start >= 1940:
        ax.set_xticks(np.arange(start, 2021, 5))
    else:
        ax.set_xticks(np.arange(start, 2021, 10))
    ax.set_xlim((start-2, 2022))

    #     Name, Lower Limit, Upper Limit, Column, Color
    props = [['Snow', '', 0, 0,  df.sn, 'C7'],
             ['Rain', '', 0, 0, df.rn, 'C2'],
             ['Frigid', '(< -15°C)', -100, -15, df.tmx, 'C1'],
             ['Freezing', '(-15–0)', -15, 0, df.tmx, 'C5'],
             ['Cold', '(0–15)', 0, 15, df.tmx, 'C4'],
             ['Cool', '(15–23)', 15, 23, df.tmx, 'C3'],
             ['Warm', '(23–30)', 23, 30, df.tmx, 'C0'],
             ['Hot', '(≥30)', 30, 100, df.tmx, 'C6']]
    props = [props[i] for i in use]

    # Make a new dataframe starting at the desired location, and make
    # a column with the correct date, but year as 2016, for plotting
    d = dt.datetime(start,1,1)
    ix = df.index.get_loc(d)
    tf = df.iloc[ix:,:].copy()
    tf['Now'] = tf.index
    tf['Now'] = tf['Now'].apply(lambda t: t.replace(year=2016))

    # make a separate frames for wet and dry days
    cn = tf.columns[4] # dry column name (Max Temp)
    # get main column names (-1 is last column, i.e. 'Now')
    mcol = tf.columns[[df.tmx] + df.precips + [-1]]
    precip = tf.columns[df.pr]
    tf.loc[np.isnan(tf[precip]), precip] = 0  # set rows without data to dry
    dryf = tf.loc[tf[precip]==0, mcol]
    wetf = tf.loc[tf[precip]>0, mcol]

    # Just select the rows that meet the criteria, then plot that row's
    # Year vs Now (Now being the date moved to 2016).
    handles = []  # create legend manually
    for name, r, ll, ul, col, c in props:
        cn = tf.columns[col]
        if col in df.precips:
            sf = wetf.loc[wetf[cn]>0]
        else:
            sf = dryf.loc[dryf[cn]>=ll]
            sf = sf.loc[sf[cn]<ul]
        ax.plot(sf.index.year, sf.Now, '_', color=c, alpha=1.0, markersize=4,
                label='')
        line = mpatches.Patch(color=c, label=' '.join([name,r]))
        handles.append(line)


    # Annotate chart
    plt.title('Precipitation (Rain, Snow) or \n'
              'Dry Days (by Daily High Temperature Range) in '+ df.city)
    ax.legend(loc='upper left', ncol=4, markerscale=3, handles=handles,
              bbox_to_anchor=(0, -0.04), handlelength=0.8, fontsize='small')
    at.Attribute(va='below', source=st.source)

    # Add second y-axis
    at.AddYAxis(ax, month=True)
    plt.show()
    return

def DayCountPlot(df, use = [0,1,2,3,4,5,6,7], column=None, style='fill',
                 trendonly=False, **kwargs):
    """Go through all data and plot what the weather was like for each day.

    Parameters
    ----------
    df : WxDF
        object containing daily data for a location. Can use a
        pandas.DataFrame if df.city comtains the name of the city.
    use : list of int default [0,1,2,3,4,5,6,7]
        Data to plot.
    column : int [4 | 6 | 8] default 4
        Which data column to use, (high, low, mean).
    style : ['fill' | 'stack' | 'line'] default 'fill'
        Style of plot to make. 'fill' fills the line to the baseline. 'stack'
        makes a stack plot where the areas add to 100%. 'line' has no fill
        and just shows the data.
    trendonly : boolean default False
        True if only the trend line is needed. The style keyword determines
        how it will look.
    kwargs : **dict
        keywords to pass to the smoothing function that override the default
        values. Supply 'trend=None' or 'trend=False' if no trend line desired.
        See Smoothing.Smooth or the trendDefault definition for explanation.

    Note
    ----
    Colours are determined by the stylesheet
    """
    trend = _GetTrendArgs(**kwargs)
    useTrend = bool(trend['trend'])
    if not column: column = df.tmx  # set default value
    tfilt = Texture()

    fig = plt.figure(df.city+'_DayCount')
    fig.clear()
    ax = fig.add_subplot(111)

    sFill = sLine = sStack = False
    if style == 'line': sLine = True
    elif style == 'stack': sStack = True
    else: sFill = True
    #     Name, Lower Limit, Upper Limit, Column, Color
    props = [['Snow', '', 0, 0,  df.sn, 'C7'],
             ['Rain', '', 0, 0, df.rn, 'C2'],
             ['Frigid', '(< -15°C)', -100, -15, df.tmx, 'C1'],
             ['Freezing', '(-15–0)', -15, 0, df.tmx, 'C5'],
             ['Cold', '(0–15)', 0, 15, df.tmx, 'C4'],
             ['Cool', '(15–20)', 15, 20, df.tmx, 'C3'],
             ['Warm', '(20–30)', 20, 30, df.tmx, 'C0'],
             ['Hot', '(≥30)', 30, 100, df.tmx, 'C6']]
    ct = {df.tmx: 'High',
          df.tmn: 'Low',
          df.tav: 'Mean'}
    props = [props[i] for i in use]
    cmap = {}  # colour map
    tmap = {}  # text values (label and range)
    [cmap.update({p[0]:p[5]}) for p in props]
    [tmap.update({p[0]:' '.join([p[0], p[1]])}) for p in props]

    # make a separate frames for wet and dry days
    cn = df.columns[column] # dry column name (Max Temp)
    precip = list(df.columns[df.precips])
    dryf = df.loc[df[precip[-1]]==0, [cn]]
    wetf = df.loc[df[precip[-1]]>0, precip]
    if sStack:
        # For days with both rain and snow, zero the lesser amount. This makes
        # stack total closer to 365.
        rain = df.columns[df.rn]
        snow = df.columns[df.sn]
        wetf.loc[wetf[rain]>=wetf[snow], snow] = 0
        wetf.loc[wetf[rain]< wetf[snow], rain] = 0

    x = list(range(df.index[0].year, df.index[-1].year+1))
    data = pd.DataFrame(index=x, dtype=int)
    colors = []
    labels = []
    for name, r, ll, ul, col, c in props:
        if col == df.tmx:
            # use provided column, not the column in props
            cn = df.columns[column]
        else:
            cn = df.columns[col]
        if col in df.precips:
            sf = wetf.loc[wetf[cn]>0, [cn]]
        else:
            sf = dryf.loc[dryf[cn]>=ll, [cn]]
            sf = sf.loc[sf[cn]<ul]
        gr = sf.groupby(sf.index.year).count()
        data[name] = gr[cn]
        data.loc[np.isnan(data[name]), name] = 0
        colors.append(c)
        labels.append(' '.join([name,r]))

    if useTrend or trendonly:
        if not trend: trend='wma'  # set default trend type if not given
        tf = pd.DataFrame(index=data.index)
        for t in data:
            tf[t] = sm.Smooth(data[t], trend)

    if trendonly:
        # Use the trend data instead of actual data
        useTrend = False
        data = tf

    # Create legend entries manually
    handles = []
    def AddLegend(c, t):
         # add a legend entry for this color
        line = mpatches.Patch(color=c, label=t)
        handles.append(line)

    sums = data.sum()  # sort by total area
    sums.sort_values(inplace=True, ascending=False)
    plotOrd = list(sums.index)

    if sFill:
        # Get plot order
        fa = 0.75
#        if trend:
#            fa = 0.15
        for p in plotOrd:
            ax.fill_between(data.index, data[p].values,
                            color=cmap[p], alpha=fa, label='',
                            agg_filter=tfilt)
            AddLegend(cmap[p], tmap[p])
        if useTrend:
            for p in plotOrd:
                ax.plot(tf.index, tf[p].values, lw=3.0,
                        color=cmap[p], alpha=1.0, label='')
    elif sStack:
        ax.stackplot(data.index, data.values.T,
                      colors=colors, alpha=0.6, labels=labels)
        [AddLegend(c, t) for c, t in zip(colors, labels)]
        if useTrend:
            sf = pd.DataFrame(index=tf.index)
            sf['sum']=0
            for p in tf:
                sf['sum'] += tf[p]
                ax.plot(sf['sum'].index, sf['sum'].values.T,
                        color = cmap[p], label='')
    elif sLine:
        for p in plotOrd:
            AddLegend(cmap[p], tmap[p])
            if not useTrend:
                if trendonly:
                    ax.plot(data.index, data[p].values, '-',
                                    color=cmap[p], alpha=st.ta, lw=st.tlw,
                                    label='')
                else:
                    ax.plot(data.index, data[p].values, '-',
                                    color=cmap[p], alpha=st.ta, lw=st.dlw,
                                    label='')
            else:
                ax.plot(data.index, data[p].values, '-',
                                color=cmap[p], alpha=st.da+.1, lw=st.dlw,
                                label='')
                ax.plot(tf.index, tf[p].values,
                    color=cmap[p], alpha=st.ta, lw = st.tlw,
                    label='')
    else:
        # do nothing by default since might not be wanted.
        pass

    # Annotate chart
    plt.title('Precipitation (Rain, Snow) or \n'
              'Dry Days (by Daily ' + ct[column] +
              ' Temperature Range) in '+ df.city)
    ax.set_ylabel('Number of Days per Year')
    ax.legend(handles=handles, loc='upper left', ncol=4, markerscale=3,
              bbox_to_anchor=(0, -0.04), handlelength=0.8, fontsize='small')
    at.Attribute(va='below', source=st.source)

    # Add second y-axis with percentages on right
    ax2, pad = at.AddYAxis(ax, percent=365)
    ax2.set_ylabel('Percent of Year')
    fig.show()

def DayThreshPlot(df, cols=None, thresh=0.0, above=True, **kwargs):
    """Count the days above or below a threshold.

    df : WxDF
        object containing daily data for a location. Can use a
        pandas.DataFrame if df.city comtains the name of the city.
    cols : list of int default [df.tmx, df.tmn]
        columns to plot.
    thresh : float default 0.0
        Threshold to test.
    above : bool default True
        Count days above threshold (warmer) or below.
        Includes threshold value.
    kwargs : **dict
        keywords to pass to the smoothing function that override the default
        values. See Smoothing.Smooth or the trendDefault definition for
        explanation.
    """

    trend = _GetTrendArgs(**kwargs)
    if not cols: cols = [df.tmx, df.tmn]
    if type(cols) != list:
        cols = [cols]
    cns = df.columns[cols]
    fig = plt.figure(df.city+'_Threshold_'+str(thresh))
    fig.clear()
    ax = fig.add_subplot(111)
    for cn in cns:
        # Get the count of days above/below threshold, grouped by year
        if above:
            ys = df.loc[df[cn]>=thresh, cn]
        else:
            ys = df.loc[df[cn]<=thresh, cn]
        ys = ys.groupby(ys.index.year).count()
        if len(ys)<2:  # don't plot if no data returned
            print('No data for '+cn)
            continue
        ax.plot(ys, lw=st.dlw, color=st.colors[cn], label='')
        tf = sm.Smooth(ys, trend)
        ax.plot(tf, lw=st.tlw, color=st.colors[cn], alpha=st.ta, label=cn)
        at.AddRate(tf.loc[1990:], label='{:.2} days/decade')
    # Set up title from possible options
    if above:
        rtxt = "At or Above"
    else:
        rtxt = "At or Below"
    if cols[0] in df.temps:
        ttxt = 'Temperature'
        utxt = '°C'
    elif cols[0]==df.sn:
        ttxt = 'Snow'
        utxt = ' cm/day'
    else:
        ttxt = 'Rain'
        utxt = ' mm/day'
    title = " ".join([df.city, 'Days with', ttxt, rtxt, str(thresh)])+utxt
    ax.set_title(title)
    at.Attribute(source=st.source, ha='left')
    plt.legend()
    ax2, pad = at.AddYAxis(ax)
    ax.set_ylabel('Days')
    plt.show()


def TemperatureCountPlot(df, use = [0,1,2,3,4,5], column=None, style='fill',
                trendonly=False, **kwargs):
    """Count the days in each temperature range. Plot in various formats.

    Parameters
    ----------
    df : WxDF
        object containing daily data for a location. Can use a
        pandas.DataFrame if df.city comtains the name of the city.
    use : list of int default [0,1,2,3,4,5,6,7]
        Data to plot. (Frigid, Freezing, Cold, Cool, Warm, Hot)
    column : int [4 | 6 | 8] default 4
        Which data column to use, (high, low, mean).
    style : ['fill' | 'stack' | 'line'] default 'fill'
        Style of plot to make. 'fill' fills the line to the baseline. 'stack'
        makes a stack plot where the areas add to 100%. 'line' has no fill
        and just shows the data.
    trendonly : boolean default False
        True if only the trend line is needed. The style keyword determines
        how it will look.
    kwargs : **dict
        keywords to pass to the smoothing function that override the default
        values. Supply 'trend=False' if no trend line is desired.
        See Smoothing.Smooth or the trendDefault definition for explanation.
    """
    trend = _GetTrendArgs(**kwargs)
    useTrend = bool(trend['trend'])
    if not column: column = df.tmx  # set default
    tfilt = Texture()
    ct = {df.tmx: 'High',
          df.tmn: 'Low',
          df.tav: 'Mean'}  # text used in title
    city = df.city

    fig = plt.figure(df.city+'_TemperatureCount')
    fig.clear()
    ax = fig.add_subplot(111)

    sFill = sLine = sStack = False
    if style == 'line': sLine = True
    elif style == 'stack': sStack = True
    else: sFill = True
    #     Name, Lower Limit, Upper Limit, Column, Color
    props = [['Frigid', '(< -15°C)', -100, -15, 'C1'],
             ['Freezing', '(-15–0)', -15, 0, 'C5'],
             ['Cold', '(0–15)', 0, 15, 'C4'],
             ['Cool', '(15–22)', 15, 22, 'C3'],
             ['Warm', '(22–30)', 22, 30, 'C0'],
             ['Hot', '(≥30)', 30, 100, 'C6']]
    props = [props[i] for i in use]  # only include provided columns
    cmap = {}  # colour map
    tmap = {}  # text values (label and range)
    [cmap.update({p[0]:p[4]}) for p in props]
    [tmap.update({p[0]:' '.join([p[0], p[1]])}) for p in props]

    cn = df.columns[column] # column name
    df = df[cn]
    # Create empty series for when all df data is outside a test case
    dfmx = df.max()
    dfmn = df.min()
    dfz = pd.Series(index=np.arange(df.index[0].year,
                                       df.index[-1].year + 1))

    x = list(range(df.index[0].year, df.index[-1].year+1))
    data = pd.DataFrame(index=x, dtype=int)
    colors = []
    labels = []
    for name, r, ll, ul, c in props:
        if (ll > dfmx) or (ul < dfmn):  # handle case where no data results
            gr = dfz
            gr[:] = 0
        else:
            sf = df[(df>=ll) & (df<ul)]
            gr = sf.groupby(sf.index.year).count()
        data[name] = gr
        data.loc[np.isnan(data[name]), name] = 0  # set nan to 0
        colors.append(c)
        labels.append(' '.join([name,r]))

    if useTrend or trendonly:
        # set default trend type if not given
        if not useTrend: trend['trend']=trendDefault['trend']
        tf = pd.DataFrame(index=data.index)
        for t in data:
            tf[t] = sm.Smooth(data[t], trend)

    if trendonly:
        # Use the trend data instead of actual data
        useTrend = False
        data = tf

    # Create legend entries manually
    handles = []
    def AddLegend(c, t):
         # add a legend entry for this color
        line = mpatches.Patch(color=c, label=t)
        handles.append(line)

    sums = data.sum()  # sort by total area
    sums.sort_values(inplace=True, ascending=False)
    plotOrd = list(sums.index)

    if sFill:
        # Get plot order
        fa = 0.75
        if useTrend:
            fa = 0.15
        for p in plotOrd:
            ax.fill_between(data.index, data[p].values,
                            color=cmap[p], alpha=fa, label='',
                            agg_filter=tfilt)
            AddLegend(cmap[p], tmap[p])
        if useTrend:
            for p in plotOrd:
                ax.plot(tf.index, tf[p].values, lw=3.0,
                        color=cmap[p], alpha=1.0, label='')
    elif sStack:
        ax.stackplot(data.index, data.values.T,
                      colors=colors, alpha=0.6, labels=labels)
        [AddLegend(c, t) for c, t in zip(colors, labels)]
        if useTrend:
            sf = pd.DataFrame(index=tf.index)
            sf['sum']=0
            for p in tf:
                sf['sum'] += tf[p]
                ax.plot(sf['sum'].index, sf['sum'].values.T,
                        color = cmap[p], label='')
    elif sLine:
        for p in plotOrd:
            AddLegend(cmap[p], tmap[p])
            if not useTrend:
                if trendonly:
                    ax.plot(data.index, data[p].values, '-',
                                    color=cmap[p], alpha=st.ta, lw=st.tlw,
                                    label='')
                else:
                    ax.plot(data.index, data[p].values, '-',
                                    color=cmap[p], alpha=st.ta, lw=st.dlw,
                                    label='')
            else:
                ax.plot(data.index, data[p].values, '-',
                                color=cmap[p], alpha=st.da+.1, lw=st.dlw,
                                label='')
                ax.plot(tf.index, tf[p].values,
                    color=cmap[p], alpha=st.ta, lw = st.tlw,
                    label='')
    else:
        # do nothing by default since might not be wanted.
        pass

    # Annotate chart
    plt.title('Count of Days by\n'
              'Daily ' + ct[column] + ' Temperature Range in '+ city)
    ax.set_ylabel('Number of Days per Year')
    ax.legend(handles=handles, loc='upper left', ncol=3, markerscale=3,
              bbox_to_anchor=(0, -0.04), handlelength=0.8, fontsize='small')
    at.Attribute(va='below', source=st.source)

    # Add second y-axis with percentages on right
    ax2, pad = at.AddYAxis(ax, percent=365)
    ax2.set_ylabel('Percent of Year')
    fig.show()

def WarmPlot(df, high=0, low=0, **kwargs):
    """
    Plot the length of the warm season over time.

    Parameters
    ----------
    df : WxDF
        object containing daily data for a location. Can use a
        pandas.DataFrame if df.city comtains the name of the city.
    high : float default 0
        Crossover temperature for the daily high. Set to None to remove it.
    low : float default 0
        Crossover temperature for the daily low. Set to None to remove it.
    kwargs : **dict
        keywords to pass to the smoothing function that override the default
        values. See Smoothing.Smooth or the trendDefault definition for
        explanation.
    """

    trend = _GetTrendArgs(**kwargs)
    longTrend = _GetTrendArgs(trend, size=61)
    cols = [df.tmx, df.tmn]
    af = df.iloc[:,cols].copy()
    maxc, minc = df.columns[cols]
    dy = ' Day'
    tr = ' Trend'
    ny = pd.Timestamp(year=2016, month=1, day=1)
    for c in [maxc, minc]:
        af[c] = sm.Smooth(df[c], longTrend)
    by = af.loc[af.index.dayofyear < 182]  # beginning of year
    ey = af.loc[af.index.dayofyear > 182]  # end of year
    wby = by.groupby(by.index.year).mean()  # just getting the proper index
    wey = wby.copy()
    for f, h in zip([wby, wey], [by, ey]):
        for c, lim in zip([maxc, minc], [high, low]):
            # get date for each year where it is just above freezing
            if lim is None: continue
            # get dates above the limit
            temp = h.loc[h[c]>lim, [c]]
            # get the remaining date with lowest temp for each year
            f[c+dy] = temp.groupby(temp.index.year).idxmin()
            # change year to 2016 for plotting
            f[c+dy] = f[c+dy].apply(lambda x: x.replace(year=2016))
            a = f[c+dy].apply(lambda x: x.dayofyear)
            a = sm.Smooth(a, trend)
            # convert dayofyear to dates
            f[c+tr] = a.apply(lambda x: ny + pd.to_timedelta(x-1, unit='d'))
    # Set up plot
    majorFmt = mdates.DateFormatter('%b %d')
    minorFmt = mdates.DateFormatter('')
    majorFmtRt = mdates.DateFormatter('%d %b')
    xlim = (wby.index.min()-5, wby.index.max()+5)
    xticks = np.arange((xlim[0]//10*10), ((xlim[1]//10+1)*10), 10)

    fig = plt.figure(df.city+'_Warm')
    fig.clear()
    title = "Average Daily Temperatures Crossing Threshold for " + df.city
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.01, wspace=0.1,
                        left=0.08, right=0.92,
                        bottom=0.05, top=0.95)
    ax0 = fig.add_subplot(2, 1, 2)  # bottom chart
    ax1 = fig.add_subplot(2, 1, 1)

    ax1.spines['bottom'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop='off')  # don't put tick labels at the top
    ax0.xaxis.tick_bottom()
    ax1.tick_params(axis='x', which='major', color=(0,0,0,0))

    # ax0 is on the bottom and should hold the beginning year results (wby)
    # Set Y Label positions and formats
    for ax, f, mxc, mnc in zip([ax0, ax1], [wby, wey],
                               [maxc, minc], [minc, maxc]):
        # find max and min value for each axes
        mx = pd.Timestamp('2016-01-01')
        mn = pd.Timestamp('2016-12-31')
        cols = f.columns
        for c in [mxc+dy, mnc+dy]:
            if c not in cols: continue
            mx = max(mx, f[c].max())
            mn = min(mn, f[c].min())
        ax.set_ylim(mn, mx)
        # locators must be declared separately for each axes but
        # the format can be reused.
        ax.yaxis.set_major_locator(mdates.DayLocator(range(5,32,5)))
        ax.yaxis.set_minor_locator(mdates.DayLocator())
        ax.yaxis.set_major_formatter(majorFmt)
        ax.yaxis.set_minor_formatter(minorFmt)
        ax.set_xticks(xticks)
        ax.set_xlim(xlim[0], xlim[1])
    for ax, f in zip([ax0, ax1], [wby, wey]):
        for c, co, lim in zip([maxc, minc], ['C0', 'C1'], [high, low]):
            if lim is None: continue
            ax.plot(f[c+tr], co+'-', lw=st.tlw, alpha=st.ta)
            ax.plot(f[c+dy], co+'o-', lw=st.dlw, alpha=st.da)

    # Create legend entries manually
    handles = []
    for c, t, lim in zip(['C0', 'C1'],
                         ['Daily High', 'Daily Low'],
                         [high, low]):
        if lim is None: continue
        line = mlines.Line2D([], [], color=c,
                             alpha=1.0, lw=st.tlw,
                             label=(t+' Crossing {:.1f} °C').format(lim))
        handles.append(line)
    plt.legend(handles=handles, loc=2)
    at.Attribute(ax=ax0, source=st.source, ha='left')

    # Add labels on right
    ax2 = ax0.twinx()
    ax3 = ax1.twinx()
    for ax, axo in zip([ax2, ax3], [ax0, ax1]):
        ax.grid(False) # is sitting on top of lines
        ax.set_yticks(axo.get_yticks())
        ax.set_ylim(axo.get_ylim())
        ax.yaxis.set_major_locator(axo.yaxis.get_major_locator())
        ax.yaxis.set_minor_locator(axo.yaxis.get_minor_locator())
        ax.yaxis.set_major_formatter(majorFmtRt)
        ax.yaxis.set_minor_formatter(minorFmt)
        ax.spines['right'].set_alpha(0)

    plt.show()

def WarmDaysPlot(df, **kwargs):
    """
    Plot the length of the warm season over time.

    Parameters
    ----------
    df : WxDF
        object containing daily data for a location. Can use a
        pandas.DataFrame if df.city comtains the name of the city.
    kwargs : **dict
        keywords to pass to the smoothing function that override the default
        values. See Smoothing.Smooth or the trendDefault definition for
        explanation.
    """

    trend = _GetTrendArgs(**kwargs)
    longTrend = _GetTrendArgs(trend, size=61)  # use for longer trends
    cols = [df.tmx, df.tmn]
    af = df.iloc[:,cols].copy()
    maxc, minc = df.columns[cols]
    dy = ' Day'
    tr = ' Trend'
    for c in [maxc, minc]:
        af[c] = sm.Smooth(df[c], longTrend)
    by = af.loc[af.index.dayofyear < 182]  # beginning of year
    ey = af.loc[af.index.dayofyear > 182]  # end of year
    wby = by.groupby(by.index.year).mean()  # just getting the proper index
    wey = wby.copy()
    diff = wby.copy()
    cy=wby.copy()  # count of all days above freezing
    xlim = (wby.index.min()-5, wby.index.max()+5)
    xticks = np.arange((xlim[0]//10*10), ((xlim[1]//10+1)*10), 10)
    for c in [maxc, minc]:
        for f, h in zip([wby, wey], [by, ey]):
            # get date for each year where it is just above freezing
            temp = h.loc[h[c]>0, [c]]
            f[c+dy] = temp.groupby(temp.index.year).idxmin()
            f[c+dy] = f[c+dy].apply(lambda x: x.dayofyear)
        diff[c+dy] = wey[c+dy] - wby[c+dy]
        diff[c+tr] = sm.Smooth(diff[c+dy], trend)
        # Collect data on total days, which might be useful later
        temp = df.loc[df[c]>0, [c]]
        cy[c+dy] = temp.groupby(temp.index.year).count()
        cy[c+tr] = sm.Smooth(cy[c+dy], trend)

    fig = plt.figure(df.city+'_WarmCount')
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_xticks(xticks)
    ax.set_xlim(xlim[0], xlim[1])
    for c, co, l in zip([maxc, minc], ['C0', 'C1'],
                        ['Daily High above 0°C', 'Daily Low above 0°C']):
        ax.plot(diff[c+dy], 'o-', color=co, lw=st.dlw, alpha=st.da, label='')
        ax.plot(diff[c+tr], '-', color=co, lw=st.tlw, alpha=st.ta,
                label=l)
        #ax.plot(cy[c+dy], '.', color=co, lw=st.dlw, alpha=st.da, label='')
        #ax.plot(cy[c+tr], '-', color=co, lw=st.dlw, alpha=st.da, label='')
        at.AddRate(diff[c+tr].loc[1970:], label='{:.2} days/decade')

    plt.title('Length of Period With Average Temperature\n'
              'above Freezing for ' + df.city)
    plt.ylabel('Days')
    plt.legend()
    at.Attribute(source=st.source)
    at.AddYAxis(ax)
    plt.show()

def SnowPlot(df, **kwargs):
    """
    Go through all data and plot first and last day of snow for the year.

    Parameters
    ----------
    df : WxDF
        object containing daily data for a location. Can use a
        pandas.DataFrame if df.city comtains the name of the city.
    """
    trend = _GetTrendArgs(**kwargs)
    shortTrend = _GetTrendArgs(trend, size=15)
    # set up data for each set of records:
    # [Name, df column, mark color and format, zorder]
    # Create list of daily records. Use 2016 as reference year (leap year)
    col = df.sn
    cn = df.columns[col]
    df = df.loc[df[cn]>0]  # get days with snow
    # put day of year with snow
    af = pd.Series(data=df.index.dayofyear, index=df.index)
    dy = ' Day'
    tr = ' Trend'
    ny = pd.Timestamp('2016-01-01')  # start of year
    fig = plt.figure(df.city+'_Snow')
    fig.clear()
    ax = fig.add_subplot(111)
    # af, by, ey are series, not dataframes
    by = af[af < 182]  # beginning of year
    ey = af[af > 182]  # end of year
    # create a dataframe to put series into
    wby = pd.DataFrame(index=list(range(df.index[0].year,
                                        df.index[-1].year+1)))
    wey = wby.copy()
    for f, h in zip([wby, wey], [by, ey]):
        # get latest or earliest date for each year where it snowed
        gr = h.groupby(h.index.year)
        if h is by:
            f[cn+dy] = gr.idxmax()
        else:
            f[cn+dy] = gr.idxmin()
        f[cn+dy] = f[cn+dy].apply(lambda x: x.replace(year=2016))
        a = f[cn+dy].apply(lambda x: x.dayofyear)
        a = sm.Smooth(a, shortTrend)
        f[cn+tr] = a.apply(lambda x: ny + pd.to_timedelta(x-1, unit='d'))
        ax.plot(f[cn+dy], 'o-', color='C5',
                 linewidth=st.dlw,
                 alpha=st.ta)
        plt.plot(f[cn+tr], '-', linewidth=st.tlw, color='C1')

    plt.title('First and Last Snowfall for ' + df.city)
    plt.axis([1885, 2020, '20161231', '20160101'])
    at.Attribute(ax, ha='left', va='top', source=st.source)
    at.MonthFmt(ax)
    at.AddYAxis(ax, month=True)
    plt.show()
    return

def TopPrecipPlot(df, cols=None, lim = 10, **kwargs):
    """Plot average precipitation for top days of each year

    Parameters
    ----------
    df : pandas.DataFrame
        daily data contained in a pandas DataFrame
    cols : list of int [df.rn | df.sn | df.pr] default [df.rn, df.sn]
        precipitation column to use (rain, snow, all)
    lim : float default 10
        percentage of annual values to use for calculation
    size : int default 21
        Size of the moving average window. Larger values give smoother
        results.
    kwargs : **dict
        keywords to pass to the smoothing function that override the default
        values. See Smoothing.Smooth or the trendDefault definition for
        explanation.
    """
    trend = _GetTrendArgs(**kwargs)
    if not cols: cols = [df.rn, df.sn]
    frac = lim/100
    tmap = dict(zip(df.precips, ['Rain', 'Snow', 'Precipitation']))
    tunit = dict(zip(df.precips, ['mm', 'cm', 'mm']))
    fig = plt.figure(df.city+'_TopPrecip')
    fig.clear()
    ax = fig.add_subplot(111)
    for col in cols:
        cn = df.columns[col]
        ps = pd.Series(index=list(range(df.index[0].year,
                                           df.index[-1].year+1)))
        rs = df.loc[df[cn]>0, cn].copy()
        gs = rs.groupby(rs.index.year)
        for yr, ys in gs:
            num = int(len(ys) * frac + 0.5)
            ps[yr] = ys.nlargest(num).mean()
        ts = sm.Smooth(ps, trend)

        ax.plot(ps, lw=st.dlw, color=st.colors[col])
        ax.plot(ts, lw=st.tlw, color=st.colors[col],
                label=tmap[col]+' ('+tunit[col]+')')
        at.AddRate(ts.loc[1970:], ax=ax,
                   label='{:.2} '+tunit[col]+'/decade')

    plt.legend()
    ax.set_title('Average Precipitation of Top {0}% '
                 'Days Per Year in {1}'.format(lim, df.city))
    ax.set_ylabel('Precipitation (mm/cm per day)')
    at.Attribute(ax, ha='left', va='bottom', source=st.source)
    at.AddYAxis(ax)
    fig.show()

def StormPlot(df, cols=None, lim = 10, **kwargs):
    """Plot average total precipitation for top storms (consecutive days of
       precipitation) of each year.

    Parameters
    ----------
    df : pandas.DataFrame
        daily data contained in a pandas DataFrame
    cols : list of int [df.rn | df.sn | df.pr] default [df.rn, df.sn]
        precipitation column to use (rain, snow, all)
    lim : float default 10
        percentage of annual values to use for calculation
    kwargs : **dict
        keywords to pass to the smoothing function that override the default
        values. See Smoothing.Smooth or the trendDefault definition for
        explanation.
    """
    trend = _GetTrendArgs(**kwargs)
    if not cols: cols = [df.rn, df.sn]  # default values

    fig = plt.figure(df.city+'_Storms')
    fig.clear()
    ax = fig.add_subplot(111)

    frac = lim/100
    tmap = dict(zip(df.precips, ['Rain', 'Snow', 'Precipitation']))
    tunit = dict(zip(df.precips, ['mm', 'cm', 'mm']))
    ps = pd.Series(index=list(range(df.index[0].year,
                                    df.index[-1].year+1)))
    for col in cols:
        # Extract the required column. Use [col] instead of just col to force
        # a dataframe instead of a series.
        tf = df.iloc[:,[col]].copy()
        cn = df.columns[col]  # column name
        # Create a new column that has True where there was no precipiation
        # and False when there was. These will be treated as 1/0 later.
        tf['dry'] = tf[cn].apply(lambda x: x==0)
        # Create a new column that has the cumulative sum of the 'dry' column.
        # What happens is that this value increases when it was dry, but
        # remains constant when it wasn't (True=1). This allows grouping
        # by consecutive precipitation days, since they will have the same
        # 'c' value.
        tf['c'] = tf['sun'].cumsum()
        # Create a new frame grouped by the year and the 'c' value, with
        # days with same 'c' value (consecutive) summed together.
        storms = tf.groupby([tf.index.year, 'c']).sum()
        # Remove the days that didn't have any precipitation
        storms = storms[storms[cn]>0]
        # Now just group by year (the first level index)
        gr = storms.groupby(level=0)
        # Go through each year, find the desired fraction of values,
        # then take the mean of the top number of values. Insert this result
        # into the precipitation series by year.
        for yr, yf in gr:
            num = int(yf[cn].count() * frac + 0.5)
            ps[yr] = yf[cn].nlargest(num).mean()
        # Get the smoothed data, and plot the results.
        ts = sm.Smooth(ps, trend)

        ax.plot(ps, lw=st.dlw, color=st.colors[col])
        ax.plot(ts, lw=st.tlw, color=st.colors[col],
                label=tmap[col]+' ('+tunit[col]+')')
        at.AddRate(ts.loc[1970:], ax=ax,
                   label='{:.2} '+tunit[col]+'/decade')

    plt.legend()
    ax.set_title('Average Total Precipitation of Top {0}% '
                 'of Storms Per Year in {1}'.format(lim, df.city))
    ax.set_ylabel('Precipitation (mm/cm per storm)')
    at.Attribute(ax, ha='left', va='bottom', source=st.source)
    at.AddYAxis(ax)
    fig.show()


def MonthRangePlot(df, month=None, combine=True, **kwargs):
    """Get expected high and low temperature ranges for the supplied month.

    Parameters
    ----------
    df : pandas.DataFrame
        daily data contained in a pandas DataFrame
    month : int default None
        Desired month (1-12). The default gives current month.
    combine : boolean default True
        Combine the maximum and minimum temperatures onto one plot. Otherwise
        use two separate plots (which is easier to read).
    kwargs : **dict
        keywords to pass to the smoothing function that override the default
        values. See Smoothing.Smooth or the trendDefault definition for
        explanation.
    Note
    ----
    Uses moving average to calculate the mean temperatures, and the standard
    deviation from this.
    """

    # Approach:
    # Get monthly means of mean, high, and low temperatures
    # Calculate the standard deviation of temperatures
    # Calculate the means ± standard deviations
    # Get smoothed means and plot those
    # Get min and max values of high and low temps and plot those.

    trend = _GetTrendArgs(**kwargs)
    if month is None or month==0 or month>12:
        month = dt.date.today().month
    maxc = df.columns[df.tmx] # max:0/4, min:1/6, avg:2/8
    minc = df.columns[df.tmn]
    avgc = df.columns[df.tav]
    umaxc = 'umaxc'
    lmaxc = 'lmaxc'
    uminc = 'uminc'
    lminc = 'lminc'
    uavgc = 'uavgc'
    lavgc = 'lavgc'
    # just use year, max, min, avg temps for desired month
    df = df.loc[df.index.month==month, [maxc, minc, avgc]]
    # Get rid of rows that have 'nan' values
    df.dropna(inplace=True)

    gb = df.groupby(df.index.year)
    sf = gb.std()
    af = gb.mean()  # mean
    mx = gb.max()  # max, highest value above mean
    mn = gb.min()  # min, lowest value below mean

    # calculate temperature ranges
    for cr, c in zip([umaxc, uminc, uavgc], [maxc, minc, avgc]):
        af[cr] = af[c] + sf[c]
    for cr, c in zip([lmaxc, lminc, lavgc], [maxc, minc, avgc]):
        af[cr] = af[c] - sf[c]

    afs = af.copy() # smoothed version of temps and ranges
    # Get the daily average max, min, avg temps.
    for c in af.columns:
        afs[c] = sm.Smooth(af[c], trend)

    # PLOTTING
    title = 'Temperature Range in '+df.city+' for '+ st.monthL[month]
    fig = plt.figure(df.city+'_'+st.monthS[month]+'_Range')
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
    ax0.fill_between(mx.index, mx[maxc], mn[maxc],
                     color='C0', alpha=st.ma, label='Upper/Lower Highs')
    ax1.fill_between(mx.index, mx[minc], mn[minc],
                     color='C1', alpha=st.ma, label='Upper/Lower Lows')
    ax0.fill_between(afs.index, afs[umaxc], afs[lmaxc],
                     color='C0', alpha=st.sa, label='68% Range Highs')
    ax1.fill_between(afs.index, afs[uminc], afs[lminc],
                     color='C1', alpha=st.sa, label='68% Range Lows')
    if not combine:
        at.AddRate(afs[maxc].loc[1970:], ax=ax0)
        at.AddRate(afs[minc].loc[1970:], ax=ax1)
    ax0.plot(afs[maxc], 'C0-', lw=2, alpha=st.ta, label='Average Highs')
    ax1.plot(afs[minc], 'C1-', lw=2, alpha=st.ta, label='Average Lows')
    if combine:
        ax0.plot(afs[avgc], 'C2-', lw=st.tlw, alpha=st.ta, label='Average Daily')

    # Add current available month as distinct points
    ly = af.index[-1]
    marks = ['^', 'o', 'v']
    maxvals = [mx.iloc[-1,0], af.iloc[-1,0], mn.iloc[-1,0]]
    minvals = [mx.iloc[-1,1], af.iloc[-1,1], mn.iloc[-1,1]]
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

    def best(s, i, comp):
        """Get the best point (highest or lowest) near the supplied point.

        s: series
        i: index location
        comp: 'low' | 'high'
        """
        lim = 3
        t = s.iloc[i-lim: i+lim]
        t.sort_values(inplace=True, ascending=(comp=='low'))
        return s.index.get_loc(t.index[0]), t.iloc[0]

    txt0 = ['Hottest Day', 'Coldest Day',
            'Average High', '68% Day Spread']
    txt1 = ['Hottest Night', 'Coldest Night',
            'Average Low', '68% Night Spread']
    va0 = ['top', 'bottom', 'bottom', 'top']
    va1 = ['top', 'bottom', 'top', 'bottom']
    yrs = len(afs.index)
    xt0 = [yrs*.05, yrs*.05, yrs*.4, yrs*.25]
    xt1 = [yrs*.05, yrs*.05, yrs*.4, yrs*.25]
    xt0 = [int(x) for x in xt0]  # text x locations
    xt1 = [int(x) for x in xt1]
    xp0 = [x+13 for x in xt0]  # arrow x locations
    xp1 = [x+13 for x in xt1]
    yt0 = [mid(mx[maxc],.8), mid(mn[maxc],.2),
           afs[maxc].iloc[xt0[2]]+2, afs[umaxc].iloc[xt0[3]]-1]
    yt1 = [mid(mx[minc]), mid(mn[minc],.2),
           afs[minc].iloc[xt1[2]-2], afs[lminc].iloc[xt1[3]]+1]
    yp0 = [mx[maxc].iloc[xp0[0]], mn[maxc].iloc[xp0[1]],
           afs[maxc].iloc[xp0[2]], afs[umaxc].iloc[xp0[3]]]
    yp1 = [mx[minc].iloc[xp0[0]], mn[minc].iloc[xp0[1]],
           afs[minc].iloc[xp1[2]], afs[lminc].iloc[xp1[3]]]
    # get closest peaks for annotation
    xp0[0], yp0[0] = best(mx[maxc], xp0[0], 'high')
    xp1[0], yp1[0] = best(mx[minc], xp1[0], 'high')
    xp0[1], yp0[1] = best(mn[maxc], xp0[1], 'low')
    xp1[1], yp1[1] = best(mn[minc], xp1[1], 'low')

    props = {'arrowstyle': '->',
             'edgecolor': 'k'}
    for t, v, i, y, ip, yp in zip(txt0, va0, xt0, yt0, xp0, yp0):
        x = afs.index[i]
        xp = afs.index[ip]
        ax0.annotate(t, (xp,yp), (x,y), va=v, ha='center',
                     color='darkred', size='smaller',
                     arrowprops=props)
    for t, v, i, y, ip, yp in zip(txt1, va1, xt1, yt1, xp1, yp1):
        x = afs.index[i]
        xp = afs.index[ip]
        ax1.annotate(t, (xp,yp), (x,y), va=v, ha='center',
                     color='darkblue', size='smaller',
                     arrowprops=props)
    if combine:
        x = afs.index[xt0[2]]
        y = afs[avgc].iloc[xt0[2]]
        ax0.text(x, y, 'Month Average',
                 ha='center', va='bottom', size='smaller')

    at.Attribute(source=st.source)

    at.AddYAxis(ax0)
    if not combine: at.AddYAxis(ax1)
    plt.show()

def Histograms(df, col=WxDF.tmx, months=None,
               llim=None, ulim=None, showMedian=True, **kwargs):
    """Plot histograms of values for groups of years
    Parameters
    ----------
    df : pandas.DataFrame
        daily data contained in a pandas DataFrame
    col : int default WxDF.tmx
        Which data column to use. Defaults to max temperature.
    months : list default None
        List of which months to use. Passing None will cause the entire
        year to be used.
    llim : float default None
        Lower limit of data, showing only values at or above this.
    ulim : float default None
        Upper limit of data, showing only values at or below this.
    showMedian : bool default True
        If True, shows a line demarking both median and mean value.
    showRaw : bool default False
        If True, shows the actual histogram bars as well as the smoothed
        line. Useful for identifying data irregularities.
    fuzz : bool default True
        If True, checks if measured units are bigger than recorded units,
        eg measured in Fahrenheit, but recorded in .1°C. If so, will
        spread the measured value over any bins it could possibly overlap.
    kwargs : **dict
        keywords to pass to the smoothing function that override the default
        values. See Smoothing.Smooth or the trendDefault definition for
        explanation.
    Note
    ----
    Use the 'size' keyword to adjust how the smoothed line looks.
    """
    trend = _GetTrendArgs(size=51, trend='ssa', pad='mirror')
    trend = _GetTrendArgs(trend, **kwargs)
    bw = kwargs.pop('bw', 1.0)  # bin width, with override
    scale = kwargs.pop('scale', 5) # amount of overlap on plots
    cn = df.columns[col]
    kind = {df.tmx: ['Daily High Temperature', '°C'],
            df.tmn: ['Daily Low Temperature', '°C'],
            df.tav: ['Daily Average Temperature', '°C'],
            df.rn: ['Daily Rain', 'mm/day'],
            df.sn: ['Daily Snow','cm/day'],
            df.pr: ['Daily Precipitation', 'mm/day (equivalend)']}
    # set up the year ranges to do histogram over
    cyear = df.index.year.max()+1 # last year in data
    ranges = []
    if df.baseline[1] < 1970:  # add period up to 1970, if applicable
        ranges.append(tuple(df.baseline)) # start with baseline period
        ranges.append((df.baseline[1], 1970))
    for r in range(1970, df.index[-1].year, 10):
        ranges.append((r, min(cyear, r+10)))

    pos = list(range(len(ranges)))  # position on the axes
    pos = pos[::-1] # reverse order for plotting
    tfilt = Texture()  # texture filter

    # Set up the figure
    fig = plt.figure(df.city+'_Histogram '+df.columns[col])
    fig.clear()
    ax = fig.add_subplot(111)
    title = f"{df.city} {kind[col][0]} Distribution"
    # add limits
    if llim is not None and ulim is not None:
        title += f"\nBetween {llim} and {ulim} {kind[col][1]}"
    elif llim is not None:
        title += f"\nAbove {llim}  {kind[col][1]}"
    elif ulim is not None:
        title += f"\nBelow {ulim} {kind[col][1]}"
    # add months
    if months is not None:
        tlist = []
        if llim is None and ulim is None: title += '\n'
        for m in months:
            tlist.append([', ', st.monthL[m]])
        tlist[0][0] = ' For '
        if len(months) > 1: tlist[-1][0] = ', and '
        for t in tlist: title += t[0] + t[1]
    ax.set_title(title)
    ax.set_xlabel(kind[col][1])

    # get just the desired months
    mf = df[cn].dropna()
    if months:
        mf = mf[mf.index.month.isin(months)]
    if col in df.precips:
        mf = mf[mf > 0]
    # get just the desired value ranges (for looking at edges)
    if llim is not None:
        mf = mf[mf >= llim]
    if ulim is not None:
        mf = mf[mf <= ulim]

    # Bins are centred on bin width multiples, ie centred on 0
    minb = np.round(mf.min()/bw)*bw - bw * .5
    maxb = np.round((mf.max())/bw)*bw + bw
    bins = np.arange(minb, maxb, bw, dtype=float)

    def GetDiff(s):
        """Return the most common jump between values. Used to determine if
           measured units differ from recorded units.
        """
        vc = s.value_counts() # get list of values by count
        keys = sorted(vc.index)
        diffs = np.array(keys[1:]) - np.array(keys[:-1])
        # there'll be numeric errors, so round before counting again
        ds = pd.Series(diffs.round(2)).value_counts()
        d0 = ds.index[0]
        if d0 == 0.6: diff = 5/9  # Fahrenheit
        elif d0 == 0.5: diff = 0.5
        elif d0 == 0.1: diff = 0.1
        elif len(ds.index) == 1:
            diff = d0
        else:
            d1 = ds.index[1]
            diff = (ds[d0] * d0 + ds[d1] * d1) / (ds[d0] + ds[d1])
        return diff

    def PlaceData(cts, col, s, check):
        """Creates counts of each recorded value. Also checks if measured
           amounts larger than recorded resolution, which would result in
           gaps in the bins. To avoid this, the values are spread out over
           the width of the measurement error, and allocating the
           measurement to multiple bins proportionately to the overlap.
        """
        vc = s.value_counts() # get list of values by count
        keys = sorted(vc.index)
        if check:  # check if distance between values large
            diff = GetDiff(s)
            if diff == 0.1: check = False
        if check:
            # allocate a portion of a value count to the bins the
            # value might overlap due to measurement error not aligning
            # with the bins.
            for k in keys:
                val = vc[k]
                kmin = k - diff * 0.5
                kmax = k + diff * 0.5
                bl = int((kmin - minb)//bw)  # get lowest bin
                bh = int((kmax - minb)//bw)  # get highest bin
                # get amount of overlap on bin
                rl = (1 - ((kmin - minb)%bw)) * bw/diff
                rh = ((kmax - minb)%bw) * bw/diff
                cts[col].iloc[bl] += val * rl
                cts[col].iloc[bh] += val * rh
                # now add any bins inbetween
                if bh-bl > 1:
                    ratio = bw/diff  # if ratio > 1, nothing will happen
                    for b in range(bl+1, bh):
                        cts[col].iloc[b] += val * ratio
        else:
            for k in keys:
                b = int((k - minb)//bw)  # which bin to use
                cts[col].iloc[b] += vc[k]  # add counts to bin
        return check

    # Now break up each range
    counts = pd.DataFrame(index=bins)
    medians = {} # dictionary of medians for each range
    means = {}
    doSpreading = kwargs.pop('fuzz', True)
    for r in ranges:
        # get the data for each range of years
        c = f"{r[0]}—{r[1]-1}"  # column/range name
        rs = mf[mf.index.year.isin(range(r[0], r[1]))]  # range series
        medians[c] = rs.median()
        means[c] = rs.mean()
        counts[c] = pd.Series(index=bins, data=np.zeros(len(bins)))
        if doSpreading:
            diffS = GetDiff(rs[rs.index.year==r[0]])
            diffE = GetDiff(rs[rs.index.year==(r[1]-1)])
            if diffS==0 or diffE==0: print(r)
            if diffS != diffE:
                for year in range(r[0], r[1]):
                    ys = rs[rs.index.year==year]
                    doSpreading = PlaceData(counts, c, ys, check=True)
            else:
                if len(rs)==0:
                    print(f'Missing data: {r[0]}-{r[1]}')
                doSpreading = PlaceData(counts, c, rs, check=True)
        else:
            PlaceData(counts, c, rs, check=False)
    for c in counts: counts[c] /= counts[c].sum()
    # counts now contains a normalized histogram in each column

    # put in 10 points per bin
    x = np.arange(minb, maxb, bw/10, dtype=float)
    hf = pd.DataFrame(index=x, dtype=float)  # holds histograms
    sf = hf.copy()  # holds smoothed data
    hpk = spk = 0  # for scaling plots relative to peak of all plots
    for c in counts:
        y = np.zeros(len(x))
        # add 10 points per bin
        for i in range(len(y)):
            b = int((x[i] - minb)//bw)
            y[i] = counts[c].iloc[b]
        hf[c] = y
        sf[c] = sm.Smooth(hf[c], trend)
        hpk = max(hpk, hf[c].max())
        spk = max(spk, sf[c].max())

    showRaw = kwargs.pop('showRaw', False)
    showMedian = kwargs.pop('showMedian', True)
    if showRaw: ratio = scale/hpk
    else: ratio = scale/spk
    bg = ax.get_facecolor()  # use background color for contrast
    bgcolor = (bg[0], bg[1], bg[2], 0.8)  # add lower alpha
    for c, p in zip(hf, pos):
        hf[c] = hf[c] * ratio + p
        sf[c] = sf[c] * ratio + p
        if showRaw:
            ax.fill_between(x, hf[c].values, p,
                            color=st.colors[col], zorder=20-p)
            ax.plot(hf[c], lw=2, color='grey', zorder=20-p)
        else:
            ax.fill_between(x, sf[c].values, p, color=st.colors[col],
                            zorder=20-p, agg_filter=tfilt)
        ax.plot(sf[c], lw=2, color='k', zorder=20-p)
        if showMedian:
            i = int(np.round((medians[c]-minb)*10))
            ax.vlines(x[i], p, sf[c].iloc[i], linestyle=':',
                      color=bgcolor, alpha=.5, lw=2, zorder=20-p)
            i = int(np.round((means[c]-minb)*10))
            ax.vlines(x[i], p, sf[c].iloc[i], linestyle='--',
                      color=bgcolor, alpha=.5, lw=2, zorder=20-p)

# Alternate approach using existing functions, but lacks options
#    parts = ax.violinplot(data, pos, points=100, vert=False, widths=6.0,
#                          showmeans=False, showextrema=False,
#                          showmedians=False)
#    for body in parts['bodies']:
#        paths = body.get_paths()[0]
#        mean = np.mean(paths.vertices[:, 1])
#        paths.vertices[:, 1][paths.vertices[:, 1] <= mean] = mean
#        body.set_edgecolor('black')
#        body.set_facecolor(st.colors[col])
#        body.set_lw(2)
#        body.set_alpha(1)


    ax.set_ylim(bottom=-1)
    ax.set_yticks(pos)
    tx = maxb
    if llim is None:
        tx = minb
    for r, p in zip(hf.columns, pos):
        t = plt.text(tx, p+.2, r, size='medium', zorder=30, va='bottom')
        t.set_path_effects([path_effects.Stroke(linewidth=3,
                            foreground=bgcolor), path_effects.Normal()])
    ax.tick_params(axis='y', labelcolor=(0,0,0,0))
    ax.tick_params(axis='y', color=(0,0,0,0))
    at.Attribute(ha='left', source=st.source)
    if showMedian:
        handles = []
        line = mlines.Line2D([], [], color='k', linestyle=':',
                             lw=2, label='Median')
        handles.append(line)
        line = mlines.Line2D([], [], color='k', linestyle='--',
                             lw=2, label='Mean')
        handles.append(line)
        plt.legend(handles=handles)
    plt.show()



def TestTexture(test=True):
    txt = Texture()
    fig = plt.figure('Test Texture')
    fig.clear()
    ax = fig.add_subplot(111)
    ax.fill_between([2,3], [1,1])
    ax.fill_between([0,1], [1,1], agg_filter=txt)
    plt.show()



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
    print(df)
