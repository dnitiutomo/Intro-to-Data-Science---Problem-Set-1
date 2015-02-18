# -*- coding: utf-8 -*-
"""
Created on Sun Jan 04 12:29:21 2015

@author: Daniel
"""
import numpy as np
from pandas import *
from ggplot import *
import datetime

def convert_to_day(row):
    return int(datetime.datetime.strptime(str(row['DATEn']),'%Y-%m-%d').strftime('%w'))

def get_avg_for_hour(row,avgs):
    return avgs[row['Hour']]

def get_avg_for_day(row,avgs):
    return avgs[row['Day']]    

def get_avg_for_unit(row,avgs):
    return avgs[str(row['UNIT'])]

turnstile_weather = pandas.read_csv(r'C:\Users\Daniel\Documents\OnlineLearning\Intro to DS\turnstile_data_master_with_weather.csv')
#df = pandas.read_csv(r'C:\Users\Daniel\Documents\OnlineLearning\Intro to DS\hr_year.csv')

pandas.options.mode.chained_assignment = None 
#Ridership by time of day

X = turnstile_weather.copy()
Y = X.pop('ENTRIESn_hourly')
sumhour = Y.groupby(X.Hour).mean().to_dict()
hourdf = pandas.DataFrame(sumhour.items(),columns=['Hour','ENTRIESn_hourly'])
#plot1 = ggplot(hourdf, aes('Hour', 'ENTRIESn_hourly')) + geom_histogram(stat="bar") + ggtitle("Avg Entries by Hour") + xlab("Hour") + ylab("Avg Entries")

#Ridership by Day of Week
turnstile_weather['Day'] = turnstile_weather.apply(convert_to_day,1)
X = turnstile_weather.copy()
Y = X.pop('ENTRIESn_hourly')
sumday = Y.groupby(X.Day).mean().to_dict()
daydf = pandas.DataFrame(sumday.items(),columns=['Day','ENTRIESn_hourly'])
#plot1 = ggplot(daydf, aes('Day', 'ENTRIESn_hourly')) + geom_histogram(stat="bar") + ggtitle("Avg Entries by Day") + xlab("Day") + ylab("Avg Entries")

#Ridership based on Subway Station
turnstile_weather['unit_cat'] = pandas.Categorical(turnstile_weather.UNIT).labels
X = turnstile_weather.copy()
Y = X.pop('ENTRIESn_hourly')
sumunit = Y.groupby(X.unit_cat).mean().to_dict()
unitdf = pandas.DataFrame(sumunit.items(),columns=['unit_cat','ENTRIESn_hourly'])
#plot1 =ggplot(unitdf, aes('unit_cat','ENTRIESn_hourly')) + geom_point() + ggtitle("Avg Entries by UNIT") + xlab("UNIT") + ylab("ENTRIESn_hourly")

#Ridership based on weather vairables
X = turnstile_weather.copy()
Y = X.pop('ENTRIESn_hourly')
sumweat = Y.groupby(X.precipi).mean().to_dict()
weatdf = pandas.DataFrame(sumweat.items(),columns=['precipi','ENTRIESn_hourly'])
#plot1 = ggplot(weatdf, aes('meantempi', 'ENTRIESn_hourly')) + geom_histogram(stat="bar") + ggtitle("Avg Entries by Var") + xlab("Var") + ylab("Avg Entries")
plot1 = ggplot(weatdf, aes('precipi', 'ENTRIESn_hourly')) + geom_point() + ggtitle("Avg Entries by Variable") + xlab("precipi") + ylab("Entries")
#View the entries by day and hour
#plot1 = ggplot(turnstile_weather, aes(x = 'Hour', y = 'ENTRIESn_hourly', fill='Day')) + geom_histogram()


X = turnstile_weather.copy()
Y = X.pop('ENTRIESn_hourly')
sumunit = Y.groupby(X.UNIT).mean().to_dict()
unitdf = pandas.DataFrame(sumunit.items(),columns=['UNIT','ENTRIESn_hourly'])
#plot1 = ggplot(unitdf, aes('ENTRIESn_hourly')) + geom_histogram(binwidth = 500)

print plot1
    #+ scale_colour_manual(values=cbPalette)
