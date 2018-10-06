#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 17:12:44 2018

@author: iain
"""
import matplotlib.pyplot as plt

def model_comp_plot(names, daily_rmse):
    
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    
    i = 0
    for name in names:
        plt.plot(days, daily_rmse[i,:], marker='o', label=name)
        i = i + 1
    
    plt.xlabel('Day of the week')
    plt.ylabel('Global active power prediction RMSE (kW)')
    plt.title('Forecasting model comparison')
    plt.legend()
    plt.show()