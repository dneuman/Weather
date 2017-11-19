#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weather Correlation

Looking at how beginning of year temps correlate to end of year temps.

Created on Thu Nov 16 12:53:50 2017

@author: dan
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import Weather as wx


df = wx.WxDF()
col = df.columns[8]  # mean temperature
df = df[['Year', col]]
df.dropna(inplace=True)
df['dy'] = df.index.dayofyear

plt.style.use('weather')
fig = plt.figure(17)
fig.clear()
ax = fig.add_subplot(111)

for m in range(2,13):
    date = pd.Timestamp(dt.date(2016, m, 1))
    dy = date.dayofyear
    bf = df.loc[df.dy <= dy] # beginning
    ef = df.loc[df.dy > dy]  # end
    yf = pd.DataFrame(bf[['Year',col]].groupby('Year').mean())
    yf['end'] = ef[['Year',col]].groupby('Year').mean()
    ax.plot(yf[col], yf.end, 'o')

fig.show()