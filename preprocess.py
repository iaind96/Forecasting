#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 09:40:26 2018

@author: iain
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv('./data/household_power_consumption.txt', sep=';', 
                      low_memory=False, infer_datetime_format=True, 
                      parse_dates={'datetime':[0,1]}, index_col=['datetime'])

dataset.replace('?', np.nan, inplace=True)
dataset.fillna(method='ffill', inplace=True)
dataset = dataset.astype('float32')

values = dataset.values
dataset['sub_metering_4'] = (values[:,0]*1000/60) - (values[:,4] + values[:,5] + values[:,6])

dataset.columns = map(str.lower, dataset.columns)

dataset.to_csv('./data/household_power_consumption.csv')

daily_groups = dataset.resample('D')
daily_data = daily_groups.agg({'global_active_power' : np.mean,
                               'global_reactive_power' : np.mean,
                               'voltage' : np.mean,
                               'global_intensity' : np.mean,
                               'sub_metering_1' : np.sum,
                               'sub_metering_2' : np.sum,
                               'sub_metering_3' : np.sum,
                               'sub_metering_4' : np.sum})

daily_data.drop(index=daily_data.index[0], inplace=True)
daily_data.drop(index=daily_data.index[-6:], inplace=True)

daily_data.to_csv('./data/household_power_consumption_daily.csv')

train_data = daily_data[:'1/2/2010']
test_data = daily_data['1/3/2010':]

train_data.to_csv('./data/train_data.csv')
test_data.to_csv('./data/test_data.csv')






