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
from sklearn.svm import SVR
from statsmodels.tsa.arima_model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

class TSForecaster():
    def evaluate_model(self, train, test, fold_length=7, max_training=None):
        n_folds = int(np.ceil(len(test)/fold_length))
        forecast = np.zeros((n_folds,fold_length))
        for i in range(forecast.shape[0]):
            self.fit(train)
            prediction = self._forecast(train, fold_length)
            forecast[i,:] = prediction
            train = train.append(test[i*fold_length:(i+1)*fold_length])
            if max_training is not None:
                train = train[-max_training:]
        forecast = forecast.reshape((-1,1))
        forecast = forecast[:len(test)]
        rmse, daily_rmse = self.evaluate_forecast(forecast, test)
        return rmse, daily_rmse
    
    def evaluate_forecast(self, forecast, actual):
        rmse = np.sqrt(mean_squared_error(forecast, actual))
        forecast = forecast.reshape((-1,7))
        actual = actual.values.reshape((-1,7))
        daily_rmse = np.zeros((7))
        for i in range(7):
            daily_rmse[i] = np.sqrt(mean_squared_error(forecast[:,i], actual[:,i]))
        return rmse, daily_rmse
        
    def forecast(self, data, n_steps=7):
        self.fit(data)
        forecast = self._forecast(data, n_steps=n_steps)
        dates = data.index + n_steps
        forecast = pd.Series(forecast, index=dates[-n_steps:])
        return forecast
    
    def simple_forecast(self, train, test):
        self.fit(train)
        forecast = np.zeros(len(test))
        for i in range(len(test)):
            forecast[i] = self._forecast(train, n_steps=1)
            train = train.append(test[[i]])
        dates = test.index
        forecast = pd.Series(forecast, index=dates)
        return forecast
    
    def _to_supervised_sliding(self, data):
        n_inputs = self.n_inputs
        N_obvs = len(data) - n_inputs
        X = np.zeros((N_obvs, n_inputs))
        Y = np.zeros((N_obvs))
        for i in range(N_obvs):
            X[i,:] = data[i:i+n_inputs]
            Y[i] = data[i+n_inputs]
        return X, Y
    
    def _to_supervised_chunks(self, data):
        y = data
        x = data.shift(1)
        n_inputs = self.n_inputs
        x.fillna(0, inplace=True)
        N_obvs = int(np.floor(len(data)/n_inputs))
        x, y = x[-N_obvs*n_inputs:], y[-N_obvs*n_inputs:]
        X = np.zeros((N_obvs, n_inputs))
        Y = np.zeros((N_obvs))
        for i in range(N_obvs):
            X[i,:] = x[i*n_inputs:(i+1)*n_inputs]
            Y[i] = y[(i+1)*n_inputs-1]
        return X, Y

class LTSForecaster(TSForecaster):
    
    def __init__(self, model, n_inputs=7):
        self.model = self.build_model(model)
        self.n_inputs = n_inputs
        
    def build_model(self, model):
        steps = []
        steps.append(('standardise', StandardScaler()))
        steps.append(('normalise', MinMaxScaler()))
        steps.append(('model', model))
        pipeline = Pipeline(steps=steps)
        return pipeline
     
    def fit(self, data):
        X_train, y_train = self._to_supervised_sliding(data)
        self.model = self.model.fit(X_train, y_train)
        return
    
    def _forecast(self, data, n_steps):
        prediction = np.zeros((n_steps))
        n_inputs = self.n_inputs
        data = data.values
        for i in range(len(prediction)):
            X = data[-n_inputs:].reshape((1,n_inputs))
            prediction[i] = self.model.predict(X)
            data = np.append(data, prediction[i])
        return prediction
    
class LSTMForecaster(TSForecaster):
    def __init__(self, n_inputs=7, n_epochs=1000):
        self.model = self.build_model(n_inputs)
        self.n_inputs = n_inputs
        self.n_epochs = n_epochs
        
    def build_model(self, n_inputs):
        model = Sequential()
        model.add(LSTM(10, input_shape=(n_inputs,1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def scale_data(self, X):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        return X_scaled
        
    def fit(self, data):
        X_train, Y_train = self._to_supervised_chunks(data)
        X_scaled = self.scale_data(X_train)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        self.model.fit(X_scaled, Y_train, epochs=self.n_epochs, verbose=0)
        
    def _forecast(self, data, n_steps):
        prediction = np.zeros((n_steps))
        n_inputs = 7
        data = data.values
        for i in range(len(prediction)):
            X = data[-n_inputs:].reshape((1, n_inputs))
            X = self.scaler.transform(X)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            prediction[i] = self.model.predict(X)
            data = np.append(data, prediction[i])
        return prediction
    
class ARForecaster(TSForecaster):
    def __init__(self, n_inputs=7):
        self.n_inputs = n_inputs
    
    def fit(self, data):
        freq = data.index.inferred_freq
        model = ARIMA(data, order=(self.n_inputs,0,0), freq=freq)
        self.model = model.fit(disp=False)
        self.training_data = data
        return
    
    def _forecast(self, data, n_steps):
        start_idx = len(self.training_data)
        end_idx = len(self.training_data)+(n_steps-1)
        prediction = self.model.predict(start_idx, end_idx)
        return prediction
        
class NTSForecaster(TSForecaster):    
    def fit(self, *args, **kwargs):
        return
     
class DPForecaster(NTSForecaster):   
    def _forecast(self, data, n_steps):
        prediction = np.array(data[-1]).repeat(n_steps)
        return prediction
    
class WPForecaster(NTSForecaster):  
    def _forecast(self, data, n_steps):
        prediction = np.zeros((n_steps))
        for i in range(len(prediction)):
            prediction[i] = data[-7+i]
        return prediction
    
class YAForecaster(NTSForecaster):
    def _forecast(self, data, n_steps):
        prediction = np.array(data[-364:-364+n_steps])
        return prediction
    
class RAForecaster(NTSForecaster):
    def __init__(self, n_inputs=7):
        self.n_inputs = n_inputs
        return
    
    def _forecast(self, data, n_steps):
        prediction = np.zeros((n_steps))
        data = data.values
        n = self.inputs
        for i in range(len(prediction)):
            prediction[i] = np.mean(data[-n:])
            data = np.append(data, prediction[i])
        return prediction   
 
    
    
default_models = {}
default_models['daily'] = DPForecaster()
default_models['weekly'] = WPForecaster()
default_models['yearly'] = YAForecaster()
default_models['ar'] = ARForecaster()
default_models['lr'] = LTSForecaster(LinearRegression())
default_models['lasso'] = LTSForecaster(Lasso())
default_models['ridge'] = LTSForecaster(Ridge())
default_models['en'] = LTSForecaster(ElasticNet())
default_models['huber'] = LTSForecaster(HuberRegressor())
default_models['lars'] = LTSForecaster(Lars())
default_models['llars'] = LTSForecaster(LassoLars())
default_models['pa'] = LTSForecaster(PassiveAggressiveRegressor(max_iter=1000, tol=1e-3))
default_models['ransac'] = LTSForecaster(RANSACRegressor())
default_models['sgd'] = LTSForecaster(SGDRegressor(max_iter=1000, tol=1e-3))
default_models['svr'] = LTSForecaster(SVR())
default_models['ra'] = RAForecaster()
default_models['lstm'] = LSTMForecaster()

def generate_models(names):
    models = {}
    for name in names:
        models[name] = default_models[name]
    return models