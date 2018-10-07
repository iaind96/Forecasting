#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:32:58 2018

@author: iain
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (LinearRegression, Lasso, Ridge, ElasticNet, 
                                  HuberRegressor, Lars, LassoLars,
                                  PassiveAggressiveRegressor, RANSACRegressor,
                                  SGDRegressor)
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
        self.fit(data)
        forecast = self._forecast(data)
        dates = data.index + 7
        forecast = pd.Series(forecast, index=dates[-7:])
        return forecast

class LTSForecaster(TSForecaster):
    
    def __init__(self, model):
        self.make_pipeline(model)
    
    def to_supervised(self, data, n_inputs):
        N_obvs = len(data) - n_inputs
        X = np.zeros((N_obvs, n_inputs))
        y = np.zeros((N_obvs))
        for i in range(N_obvs):
            X[i,:] = data[i:i+n_inputs]
            y[i] = data[i+n_inputs]
        return X, y
    
    def make_pipeline(self, model):
        steps = []
        steps.append(('standardise', StandardScaler()))
        steps.append(('normalise', MinMaxScaler()))
        steps.append(('model', model))
        self.model = Pipeline(steps=steps)
        return
     
    def fit(self, data):
        n_inputs = 7
        X_train, y_train = self.to_supervised(data, n_inputs)
        self.n_inputs = n_inputs
        self.model = self.model.fit(X_train, y_train)
        return
    
    def _forecast(self, data):
        prediction = np.zeros((7))
        n_inputs = self.n_inputs
        data = data.values
        for i in range(len(prediction)):
            X = data[-n_inputs:].reshape((1,-1))
            prediction[i] = self.model.predict(X)
            data = np.append(data, prediction[i])
        return prediction
        
class NTSForecaster(TSForecaster):    
    def fit(self, *args):
        return
     
class DPForecaster(NTSForecaster):   
    def _forecast(self, data):
        prediction = np.array(data[-1]).repeat(7)
        return prediction
    
class WPForecaster(NTSForecaster):  
    def _forecast(self, data):
        prediction = np.array(data[-7:])
        return prediction
    
class YAForecaster(NTSForecaster):
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
    
def get_models():
    models = {}
    models['daily'] = DPForecaster()
    models['weekly'] = WPForecaster()
    models['yearly'] = YAForecaster()
    models['ar'] = ARForecaster()
    models['lr'] = LTSForecaster(LinearRegression())
    models['lasso'] = LTSForecaster(Lasso())
    models['ridge'] = LTSForecaster(Ridge())
    models['en'] = LTSForecaster(ElasticNet())
    models['huber'] = LTSForecaster(HuberRegressor())
    models['lars'] = LTSForecaster(Lars())
    models['llars'] = LTSForecaster(LassoLars())
    models['pa'] = LTSForecaster(PassiveAggressiveRegressor(max_iter=1000, tol=1e-3))
    models['ranscac'] = LTSForecaster(RANSACRegressor())
    models['sgd'] = LTSForecaster(SGDRegressor(max_iter=1000, tol=1e-3))
    return models
    