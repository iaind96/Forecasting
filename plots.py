#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 17:12:44 2018

@author: iain
"""
import matplotlib.pyplot as plt

def model_comp_plot(names, rmse, daily_rmse):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    ax1.bar(range(len(names)), rmse, tick_label=names, color=color_cycle)
    ax1.set(ylabel='Forecast RMSE (kW)')
    ax1.set_title('Overall forecast RMSE')
    
    i = 0
    for name in names:
        ax2.plot(days, daily_rmse[i,:], marker='o', label=name)
        i = i + 1
    
    ax2.set_title('Per day forecast RMSE')
    ax2.legend()