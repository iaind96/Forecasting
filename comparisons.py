#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 18:39:38 2018

@author: iain
"""
import numpy as np
from plots import model_comp_plot

def compare_models(names, models, train_data, test_data):  
    rmse = np.zeros(len(names))
    daily_rmse = np.zeros((len(names),7))
    i = 0
    for name in names:
        model = models[name]
        score, scores = model.evaluate_model(train_data, test_data)
        rmse[i] = score
        daily_rmse[i,:] = scores
        scores = ', '.join(['%.3f' % s for s in scores])
        print('%s: [%.3f] %s' % (name, score, scores))
        i = i + 1
    
    model_comp_plot(names, rmse, daily_rmse)
    return rmse, daily_rmse