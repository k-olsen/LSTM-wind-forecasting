
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:08:53 2021

@author: Kira
"""

import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from scipy.interpolate import interp1d
from datetime import datetime
import math

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import sklearn.metrics


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] # gather input and output parts of the pattern
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


# Data from: https://www.kaggle.com/jorgesandoval/wind-power-generation/code
# Units of Terrawatt hours. Values are total for a given company at all of that company's stations in Germany
# Data from 23/08/2019 to 22/09/2020
#%% Read in dataset

os.chdir('/Users/Kira/OneDrive/Data Science/My_DS_Projects/Wind power forecasting')
in_file = 'Amprion.csv'

df = pd.read_csv(in_file)

count=0
for i in df.isnull().sum():
    if i==1 :
        count=1
if(count==0):
    print("No NUll Value in DF")

#%% Get data into single timeseries with datetime object for each data point

date = df.iloc[0].Date
times = df.columns

dt_string = date + ' ' + times[1]
dt = datetime.strptime(dt_string, '%d/%m/%Y %H:%M:%S')

time_list = []
data_list = []
for i in range(df.shape[0]): # Rows = days
    date = df.iloc[i].Date
    for j in range(df.shape[1]): # Columns = times
        if j == 0:
            continue # first column is date
        else:
            t = times[j]
            dt_string = date + ' ' + t
            dt = datetime.strptime(dt_string, '%d/%m/%Y %H:%M:%S')
            time_list.append(dt)
            
            data_list.append(df.iloc[i][j])

# Make into dataframe
ts = pd.DataFrame(list(zip(time_list, data_list)), columns =['datetime', 'value'])

#%% Clean up data by interpolating between 0 values

time_array = np.asarray(time_list)
data_array = np.asarray(data_list)
y = np.arange(len(data_array))

idx = np.nonzero(data_array)
interp = interp1d(y[idx],data_array[idx])

ynew = interp(y)
#%% Plot figure to show interpolation

fig, ax = plt.subplots()
plt.axhline(y = 0, c = 'lightgrey', zorder =0 )
plt.plot(ts.datetime, ts.value)
plt.scatter(ts.datetime, ts.value, s = 10, label = 'Raw Data')

plt.plot(time_array, ynew, c = 'm', lw = 1, label = 'Interpolation') # Linear Interpolation between 0-value points
plt.ylabel('Energy Production [TWh]')
plt.xlabel('Date')
plt.legend()

# Major ticks every 6 months.
fmt_half_year = mdates.DayLocator(interval=1)
ax.xaxis.set_major_locator(fmt_half_year)

# Minor ticks every month.
fmt_hr = mdates.HourLocator()
ax.xaxis.set_minor_locator(fmt_hr)

# Text in the x axis will be displayed in 'YYYY-mm' format.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%y'))

plt.xlim(datetime(2019, 8, 28, 23, 0), datetime(2019, 8, 30, 1, 0)) # To zoom in on dropped out value
plt.ylim(-5,60)

# plt.savefig('Interpolation_example.eps')
#%% Prepare data, build LSTM model, make prediction

# define input sequence
train_start = 0
train_stop = 10000# specify part of dataset to use, if you don't want full dataset
raw_seq = ynew[train_start:train_stop] 
test_start = train_stop
test_stop = 15000

n_steps = 8
n_features = 1

X, y = split_sequence(raw_seq, n_steps) # split into samples
X = X.reshape((X.shape[0], X.shape[1], n_features)) # reshape from [samples, timesteps] into [samples, timesteps, features]


# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
a = model.fit(X, y, epochs=50, verbose = True)

# Make prediction
test_seq = ynew[test_start:test_stop] # Part of dataset to predict
X2, y2 = split_sequence(test_seq, n_steps)
x_input = X2.reshape((X2.shape[0], n_steps, n_features))
yhat = model.predict(x_input, verbose= True)
yhat_list = [el for el in yhat]
#%% Plot data and prediction 

fig, ax  = plt.subplots(figsize = (10,4))
plt.plot(time_list[:test_stop], ynew[:test_stop])
plt.plot(time_list[:test_stop], ynew[:test_stop], c = 'tab:blue', label = 'Data')


# Plot all predicted values
t1 = test_start + n_steps # First predicted point is 8 points in...
t2 = t1 + len(yhat)
bb = time_list[t1:t2] 
ax.plot(bb, yhat_list, c = 'r', label = 'Prediction')

ax.set_ylabel('Energy Production [TWh]')
ax.set_xlabel('Date')
plt.legend()

# Major ticks every day
fmt_half_year = mdates.MonthLocator(interval=1)
ax.xaxis.set_major_locator(fmt_half_year)

# Minor ticks every day.
fmt_minor = mdates.DayLocator(interval=1)
ax.xaxis.set_minor_locator(fmt_minor)

# Text in the x axis will be displayed in 'YYYY-mm' format.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%y'))

true_list = ynew[t1:test_stop]
mse = sklearn.metrics.mean_squared_error(true_list, yhat_list)
rmse = math.sqrt(mse)
plt.title('RMSE: %0.2f' %(rmse))

#%% Plot Figure - Zoomed in
 
fig, ax  = plt.subplots(figsize = (4,4))
plt.plot(time_list[:test_stop], ynew[:test_stop])
plt.scatter(time_list[:test_stop], ynew[:test_stop], c = 'tab:blue', label = 'Observed Data')

# Show points used in prediction
ax.plot([time_list[t1-1], time_list[t1]], [ynew[t1-1], yhat_list[0]], c = 'r')
ax.scatter(time_list[t1], yhat_list[0], c = 'r', label = 'Prediction')
ax.plot(time_list[test_start:t1], x_input[0], c = 'y')
ax.scatter(time_list[test_start:t1], x_input[0], c = 'y', zorder = 3, label = 'Data Points Used in Prediction')


ax.set_ylabel('Energy Production [TWh]')
ax.set_xlabel('Date')
plt.legend()
ax.set_xlim(datetime(2019, 12, 5, 2, 00), datetime(2019, 12, 5, 7, 00)) # For zoom in
ax.set_ylim(20,70)

# Major ticks every day
fmt_half_year = mdates.DayLocator(interval=1)
ax.xaxis.set_major_locator(fmt_half_year)

# Minor ticks every day.
fmt_minor = mdates.HourLocator(interval=1)
ax.xaxis.set_minor_locator(fmt_minor)

# Text in the x axis will be displayed in 'YYYY-mm' format.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%y \n %H:%M'))
plt.tight_layout()

