#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:07 2018

@author: iain
"""
import pandas as pd
import numpy as np
from models import DPForecaster, WPForecaster, YOForecaster, ARForecaster  
from comparisons import compare_models  

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

models = {'daily' : DPForecaster(),
          'weekly' : WPForecaster(),
          'yearly' : YOForecaster(),
#          'ar' : ARForecaster()
          }


compare_models(['daily', 'weekly', 'yearly'], models, train_data, test_data)

    

