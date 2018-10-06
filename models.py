#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:32:58 2018

@author: iain
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

class TSForecaster():
    
    def evaluate_model(self, train, test):
        forecast = np.zeros((len(test)//7,7))
        for i in range(forecast.shape[0]):
            self.fit(train)
            prediction = self._forecast(train)
            forecast[i,:] = prediction
            train = train.append(test[i*7:(i+1)*7])
        forecast = forecast.reshape((-1,1))
        scores, score = self.evaluate_forecast(forecast, test)
        return scores, score
    
    def evaluate_forecast(self, forecast, actual):
        score = np.sqrt(mean_squared_error(forecast, actual))
        forecast = forecast.reshape((-1,7))
        actual = actual.values.reshape((-1,7))
        scores = np.zeros((7))
        for i in range(7):
            scores[i] = np.sqrt(mean_squared_error(forecast[:,i], actual[:,i]))
        return score, scores
        
    def forecast(self, data):
        self.fit()
        forecast = self._forecast(data)
#        data.index.freq = data.index.inferred_freq
        dates = data.index + 7
        forecast = pd.Series(forecast, index=dates[-7:])
        return forecast
        
class DPForecaster(TSForecaster):
    
    def fit(self, *args):
        return
    
    def _forecast(self, data):
        prediction = np.array(data[-1]).repeat(7)
        return prediction
    
class WPForecaster(TSForecaster):
    
    def fit(self, *args):
        return
    
    def _forecast(self, data):
        prediction = np.array(data[-7:])
        return prediction
    
class YOForecaster(TSForecaster):
    
    def fit(self, *args):
        return
    
    def _forecast(self, data):
        prediction = np.array(data[-364:-357])
        return prediction
    
class ARForecaster(TSForecaster):
    
    def fit(self, data):
        freq = data.index.inferred_freq
        model = ARIMA(data, order=(7,0,0), freq=freq)
        self.model = model.fit(disp=False)
        return
    
    def _forecast(self, data):
        prediction = self.model.predict(len(data), len(data)+6)
        return prediction