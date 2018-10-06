#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:32:58 2018

@author: iain
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class TSForecaster():
    
    def evaluate_model(self, train_series, test_series):
        train_vals = train_series.values
        test_vals = test_series.values
        forecast = np.zeros((len(test_vals)//7,7))
        for i in range(forecast.shape[0]):
            self.fit()
            prediction = self.make_forecast(train_vals)
            forecast[i,:] = prediction
            train_vals = np.append(train_vals, test_vals[i:(i+7)])
        forecast = forecast.reshape((-1,1))
        scores, score = self.evaluate_forecast(forecast, test_vals)
        return scores, score
        
#        train_vals = self.unpack_data(train)
#        test_vals = self.unpack_data(test)
#        train_vals = [x for x in train_vals]
#        forecast = []
#        for i in range(len(test_vals)):
#            self.fit()
#            prediction = self.make_forecast(train_vals)
#            forecast.append(prediction)
#            train_vals.append(test_vals[i,:])
#        forecast = np.array(forecast)
#        score, scores = self.evaluate_forecast(forecast, test_vals[:,:,0])
#        return score, scores
    
    def evaluate_forecast(self, forecast, actual):
        score = np.sqrt(mean_squared_error(forecast, actual))
        forecast = forecast.reshape((-1,7))
        actual = actual.reshape((-1,7))
        scores = []
        for i in range(7):
            scores.append(np.sqrt(mean_squared_error(forecast[:,i], actual[:,i])))
        return score, scores
        
        
#        scores = []
#        for i in range(actual.shape[1]):
#            mse = mean_squared_error(actual[:,i], forecast[:,i])
#            rmse = np.sqrt(mse)
#            scores.append(rmse)
#        score = np.sqrt(mean_squared_error(forecast, actual))    
#        return score, scores
        
#    def forecast(self, data, M):
#        data_vals = self.unpack_data(data)
#        forecast = self.make_forecast(data_vals)
#        dates = data.index + 7
#        forecast = pd.Series(forecast, index=dates[-7:])
#        return forecast
#        
#    def unpack_data(self,data):
#        data = data.values
#        data = np.array(np.split(data, len(data)/7))
#        return data
        
class DPForecaster(TSForecaster):
    
    def fit(self):
        return
    
    def make_forecast(self, data):
        prediction = np.array(data[-1]).repeat(7)
        return prediction
        
#        last_week = data[-1]
#        value = last_week[-1,0]
#        prediction = [value for _ in range(7)]
#        return prediction
    
class WPForecaster(TSForecaster):
    
    def fit(self):
        return
    
    def make_forecast(self, data):
        last_week = data[-1]
        return last_week[:,0]
    
class YOForecaster(TSForecaster):
    
    def fit(self):
        return
    
    def make_forecast(self, data):
        last_week = data[-52]
        return last_week[:,0]