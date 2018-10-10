#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:07:57 2018

@author: iain
"""
import pandas as pd

class DataLoader():
    def load(self):
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
        
        return train_data, test_data