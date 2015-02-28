# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 22:46:03 2015

@author: Daniel
"""

import numpy as np
import pandas
import matplotlib.pyplot as plt
import datetime

def sum_entries(row, sums):
    return sums[str(row['UNIT'])]

def day_of_week(row):
    return datetime.datetime.strptime(row['DATEn'],'%Y-%m-%d').strftime('%a')

df = pandas.read_csv(r'C:\Users\Daniel\Documents\OnlineLearning\Intro to DS\Github\turnstile_data_master_with_weather.csv')

df['Day']=df.apply(day_of_week,1)

width = 0.35

colors = ['r','b','g','c','m','y','k']

days = pandas.unique(df['Day'])
totals = []
xax = []

for i in range(0,7):
    test = df[df['Day']== days[i]]
    X = test.copy()
    y = X.pop('ENTRIESn_hourly')
    sums = pandas.Series(y.groupby(X.Hour).mean(), name = ['Entries']).tolist()
    totals.append(sums)
    xax = sums.index
    
index = np.arange(len(totals[0])) + 0.3
y_offset = np.array([0.0] * len(totals[0]))

for row in range(len(totals)):
    plt.bar(index, totals[row], width, bottom=y_offset, color=colors[row], label=days[i])  
    y_offset = y_offset + totals[row]

plt.legend(days, loc=2)
plt.title('Avg. Entries per Day by Hour')
plt.xlabel('Hours')
plt.ylabel('Entries')

plt.show()