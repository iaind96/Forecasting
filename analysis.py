#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 18:39:38 2018

@author: iain
"""
import numpy as np
from plots import model_comp_plot, plot_forecast

def compare_models(names, models, train_data, test_data):  
    rmse = np.zeros(len(names))
    daily_rmse = np.zeros((len(names),7))
    for i in range(len(names)):
        model = models[names[i]]
        score, scores = model.evaluate_model(train_data, test_data)
        rmse[i] = score
        daily_rmse[i,:] = scores
        
    print_model_comp(names, rmse, daily_rmse)
    
    model_comp_plot(names, rmse, daily_rmse)
    return rmse, daily_rmse

def compare_forecasts(names, models, train_data, test_data):
    forecasts = []
    for name in names:
        forecast = models[name].forecast(train_data)
        forecasts.append(forecast)
    plot_forecast(names, train_data, forecasts, test_data)
    
def print_model_comp(names, rmse, daily_rmse):
    for i in range(len(names)):
        scores = ', '.join(['%.3f' % s for s in daily_rmse[i,:]])
        print('%s: [%.3f] %s' % (names[i], rmse[i], scores))