#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 17:12:44 2018

@author: iain
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def model_comp_plot(names, rmse, daily_rmse): 
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    ax1.bar(range(len(names)), rmse, tick_label=names, color=color_cycle)
    ax1.set(ylabel='Forecast RMSE (kW)')
    ax1.set_title('Overall Forecast RMSE')
    
    i = 0
    for name in names:
        ax2.plot(days, daily_rmse[i,:], marker='o', label=name)
        i = i + 1
    
    ax2.set_title('Per-day Forecast RMSE')
    ax2.legend()
    fig.autofmt_xdate()
    
    return

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
        return