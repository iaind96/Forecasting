#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 18:39:38 2018

@author: iain
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  
import time
from plots import plot_forecast

def exp_status(exp_num):
    def exp_status_decorator(func):
        def wrapper(*args, **kwargs):
            print('-----------------------------------------------')
            print('EXPERIMENT ' + str(exp_num) + ' RUNNING')
            print('-----------------------------------------------')
            start_time = time.perf_counter()
            results = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            print('-----------------------------------------------')
            print('EXPERIMENT ' + str(exp_num) + ' COMPLETE. RUNTIME: {:.1f} seconds'.format(run_time))
            print('-----------------------------------------------')
            return results
        return wrapper
    return exp_status_decorator      

class Experiment1():
    def __init__(self, models, train, test, fold_length=None, max_training=None):
        self.train = train
        self.test = test
        self.fold_length = fold_length
        self.max_training = max_training
        self.models = models
        self.names = list(models.keys())
        
    @exp_status(1)
    def run_experiment(self):
        rmse = np.zeros(len(self.names))
        daily_rmse = np.zeros((len(self.names),7))
        for i, name in enumerate(self.names):
            print(name + ': EVALUATING')
            model = self.models[name]
            rmse[i], daily_rmse[i,:] = model.evaluate_model(self.train, self.test,
                                                            fold_length=self.fold_length,
                                                            max_training=self.max_training)
            print(name + ': DONE')
        results = {'rmse' : rmse, 'daily_rmse' : daily_rmse}
        self.results = results
        return results
    
    def print_results(self):
        names = self.names
        rmse = self.results['rmse']
        daily_rmse = self.results['daily_rmse']
        print('-----------------------------------------')
        print('EXPERIMENT 1 RESULTS: [rmse] per_day_rmse')
        print('-----------------------------------------')
        for i in range(len(names)):
            scores = ', '.join(['%.3f' % s for s in daily_rmse[i,:]])
            print('%s: [%.3f] %s' % (names[i], rmse[i], scores))
    
    def plot_results(self):
        names = self.names
        rmse = self.results['rmse']
        daily_rmse = self.results['daily_rmse']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
        days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
        ax1.bar(range(len(names)), rmse, tick_label=names, color=color_cycle)
        ax1.set(ylabel='Forecast RMSE (kW)')
        ax1.set_title('Overall Forecast RMSE')    
        
        for i, name in enumerate(names):
            ax2.plot(days, daily_rmse[i,:], marker='o', label=name)
        ax2.set_title('Per-day Forecast RMSE')
        ax2.legend()
        fig.autofmt_xdate()

class Experiment2():
    def __init__(self, models, train, test=None, n_steps=7):
        self.models = models
        self.names = list(models.keys())
        self.train = train
        self.test = test
        self.n_steps = n_steps
        
    @exp_status(2)
    def run_experiment(self):
        forecasts = []
        for name in self.names:
            forecast = self.models[name].forecast(self.train, n_steps=self.n_steps)
            forecasts.append(forecast)
        results = {'forecasts' : forecasts}
        self.results = results
        
    def plot_results(self):
        plot_forecast(self.names, self.train, self.results['forecasts'],
                      actual=self.actual)
        
#        names = self.names
#        prior = self.train[-21:]
#        forecast = self.results['forecasts']
#        actual = self.test
#        
#        fig, ax = plt.subplots()
#        ax.plot(prior, label='prior data')
#        
#        if not isinstance(forecast, list):
#            forecasts = []
#            forecasts.append(forecast)
#        else:
#            forecasts = forecast
#        
#        if actual is not None:
#            actual = actual[:len(forecasts[0])]
#            actual = prior[-1:].append(actual)
#            ax.plot(actual, label='actual data')
#            
#        for i in range(len(forecasts)):
#            forecasts[i] = prior[-1:].append(forecasts[i])
#            label = names[i] + ' forecast'
#            ax.plot(forecasts[i], label=label)
#        
#        ax.grid(True)
#        ax.set_title('Global Power Consumption Forecast')
#        ax.set(xlabel='Date', ylabel='Power Consumption (kW)')
#        ax.legend()
#        
#        weeks = mdates.WeekdayLocator(byweekday=1)
#        weeksFmt = mdates.DateFormatter('%d/%m/%y')
#        ax.xaxis.set_major_locator(weeks)
#        ax.xaxis.set_major_formatter(weeksFmt)    
        