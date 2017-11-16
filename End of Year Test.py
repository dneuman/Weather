#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting error versus estimation length (day of year)


Created on Fri Nov 10 19:27:07 2017

@author: dan
"""

import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import Weather as wx

plt.style.use('weather')
df = wx.WxDF()

fig = plt.figure(16)
fig.clear()
ax = fig.add_subplot(111)

yr = 2016
col = df.columns[8]

actual = df.loc[df.Year==yr,col].mean()
ax.axhline(actual, ls='--', lw=1, color='C1',
           label='Actual Average Temperature')
df['dy'] = df.index.dayofyear
data = []
p = []

for i in np.arange(2,13):
    m = int(np.floor(i))
    day = int((i-m)*30+1)
    date = pd.Timestamp(dt.date(yr, m, day))
    tf = df.loc[df.index < date, ['Year', 'dy', col]]

    dy = date.dayofyear
    # get days in full year
    fy = pd.Timestamp(dt.date(yr, 12, 31)).dayofyear

    # For all previous years, get days up to and including last day,
    # and days afterwards. Then get the sum for each year.
    bf = tf.loc[tf.dy <= dy] # beginning
    ef = tf.loc[tf.dy > dy]  # end
    yf = pd.DataFrame(bf[['Year',col]].groupby('Year').mean())
    yf['end'] = ef[['Year',col]].groupby('Year').mean()

    # The difference between beginning of year average temperature should be
    # correlated with the end of year temp, so calculate this for every year,
    # then get the stats for the differences to determine how the end of the
    # last year should end up.
    yf['diff'] = yf['end'] - yf[col]
    # Get results weighted by amount of year left
    bw = dy/fy  # beginning weight
    ew = 1.0 - bw  # end weight
    yb = yf.loc[yr, col]  # beginning temp
    const = yb * bw + yb * ew
    yf['est'] = const + yf['diff'] * ew
    yf.dropna(inplace=True)
    data.append(yf['est'].values)
    p.append(i-1)

# Make Violin Plot and customize how it looks
parts = ax.violinplot(data, p)
c = 'C0'  # color to use
for p in parts['bodies']:
    p.set_facecolor(c)
for p in ['cbars','cmaxes','cmins']:
    parts[p].set_color(c)

ax.set_title('Probability Distribution of Year Average ({})\n'
             'Based on Beginning of Year Temperature for Ottawa'.format(yr))
ax.set_ylabel('Year Mean Temperature (°C)')
ax.set_xlabel('Months of Actual Data')
ax.legend()
plt.show()

''' Note: If you only want half the curve:
v1 = ax.violinplot(data1, points=50, positions=np.arange(0, len(data1)), widths=0.85,
               showmeans=False, showextrema=False, showmedians=False)
for b in v1['bodies']:
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0],
                                              m, np.inf)
    b.set_color('r')
'''
