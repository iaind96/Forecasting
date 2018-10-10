#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 21:51:38 2018

@author: iain
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates 

def plot_forecast(names, prior, forecast, actual=None):
    prior = prior[-21:]
    fig, ax = plt.subplots()
    ax.plot(prior, label='prior data')
        
    if not isinstance(forecast, list):
        forecasts = []
        forecasts.append(forecast)
    else:
        forecasts = forecast
        
    if actual is not None:
        actual = actual[:len(forecasts[0])]
        actual = prior[-1:].append(actual)
        ax.plot(actual, label='actual data')
            
    for i in range(len(forecasts)):
        forecasts[i] = prior[-1:].append(forecasts[i])
        label = names[i] + ' forecast'
        ax.plot(forecasts[i], label=label)
    
    ax.grid(True)
    ax.set_title('Global Power Consumption Forecast')
    ax.set(xlabel='Date', ylabel='Power Consumption (kW)')
    ax.legend()
    
    weeks = mdates.WeekdayLocator(byweekday=1)
    weeksFmt = mdates.DateFormatter('%d/%m/%y')
    ax.xaxis.set_major_locator(weeks)
    ax.xaxis.set_major_formatter(weeksFmt)   