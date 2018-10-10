#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:07 2018

@author: iain
"""
from utils import DataLoader
from models import generate_models
from experiments import Experiment1, Experiment2

train_data, test_data = DataLoader().load()

max_training = None
n_steps = 7
fold_length = 7

model_subset = ['yearly', 'lr', 'ar', 'lstm']
models = generate_models(model_subset)

#exp1 = Experiment1(models, train_data, test_data, fold_length=fold_length,
#                   max_training=max_training)
#exp1.run_experiment()
#
#exp2 = Experiment2(models, train_data, test=test_data, n_steps=n_steps)
#exp2.run_experiment()

forecast = models['lstm'].simple_forecast(train_data, test_data)
