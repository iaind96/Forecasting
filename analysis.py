#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 18:39:38 2018

@author: iain
"""
import numpy as np
from plots import model_comp_plot, plot_forecast

def compare_models(names, models, train, test, fold_length=None, n_inputs=None,
                   n_training=None):  
    rmse = np.zeros(len(names))
    daily_rmse = np.zeros((len(names),7))
    for i in range(len(names)):
        model = models[names[i]]
        rmse[i], daily_rmse[i,:] = model.evaluate_model(train, test,
                                                        fold_length=fold_length,
                                                        n_inputs=n_inputs,
                                                        n_training=n_training)
        
    print_model_comp(names, rmse, daily_rmse)
    
    model_comp_plot(names, rmse, daily_rmse)
    return rmse, daily_rmse

def compare_forecasts(names, models, train, test=None, n_steps=7, n_training=None):
    forecasts = []
    for name in names:
        forecast = models[name].forecast(train, n_steps=n_steps,
                                         n_training=n_training)
        forecasts.append(forecast)
    plot_forecast(names, train, forecasts, test)
    
def print_model_comp(names, rmse, daily_rmse):
    print('============================')
    print('RESULTS: [rmse] per_day_rmse')
    print('============================')
    for i in range(len(names)):
        scores = ', '.join(['%.3f' % s for s in daily_rmse[i,:]])
        print('%s: [%.3f] %s' % (names[i], rmse[i], scores))