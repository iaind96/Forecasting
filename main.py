#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:07 2018

@author: iain
"""
import pandas as pd
import matplotlib.pyplot as plt
from models import DPForecaster, WPForecaster, YOForecaster

def summarise_scores(name, score, scores):
    scores = ', '.join(['%.3f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, scores))
    

train_data = pd.read_csv('./data/train_data.csv', low_memory=False, 
                         infer_datetime_format=True, parse_dates=True,
                         index_col=['datetime'])
test_data = pd.read_csv('./data/test_data.csv', low_memory=False,
                         infer_datetime_format=True, parse_dates=True,
                         index_col=['datetime'])

train_data = train_data['global_active_power']
test_data = test_data['global_active_power']

days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']

dpf = ('daily', DPForecaster())
wpf = ('weekly', WPForecaster())
yof = ('yearly', YOForecaster())

models = [dpf]

for name, model in models:
    score, scores = model.evaluate_model(train_data, test_data)
    summarise_scores(name, score, scores)
    plt.plot(days, scores, marker='o', label=name)
    
plt.legend()
plt.show()
    

