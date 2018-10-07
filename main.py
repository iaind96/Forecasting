#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:07 2018

@author: iain
"""
import pandas as pd
from models import get_models
from analysis import compare_models, compare_forecasts

train_data = pd.read_csv('./data/train_data.csv', low_memory=False, 
                         infer_datetime_format=True, parse_dates=True,
                         index_col=['datetime'])
test_data = pd.read_csv('./data/test_data.csv', low_memory=False,
                         infer_datetime_format=True, parse_dates=True,
                         index_col=['datetime'])

train_data.index.freq = train_data.index.inferred_freq
test_data.index.freq = test_data.index.inferred_freq

train_data = train_data['global_active_power']
test_data = test_data['global_active_power']

models = get_models()

model_subset = ['yearly', 'lr', 'ransac']

compare_models(model_subset, models, train_data, test_data)
compare_forecasts(model_subset, models, train_data, test_data)