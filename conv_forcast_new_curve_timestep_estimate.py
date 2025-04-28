
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.preprocessing import StandardScaler
from scipy import signal
import random

from sklearn.model_selection import train_test_split


def create_exp_curve(y_start, y_stop, n_points, k):

    # Create x array (normalized from 0 to 1)
    x = np.linspace(0, 1, n_points)

    # Exponential rise
    y = y_start + (y_stop - y_start) * (1 - np.exp(-k * x))
    return y


def test_model(scaler, index, model, sample_profile, prediction_days, plot, forcast_err, err_count=None, pred_amount=25):

    input = scaler.transform(sample_profile[index-prediction_days:index].reshape(-1,1))
    future = sample_profile[index:index+pred_amount]
    #input = sample_profile[index-prediction_days:index].reshape(-1,1)

    lst = []
    lst.append(input.reshape(prediction_days,1))
    lst = np.array(lst)
    pred = model.predict(lst)
    #pred = pred[0]
    input = pred

    #input = input.reshape(-1,1)
    input = scaler.inverse_transform(input.reshape(-1,1))

    input = input.reshape(pred_amount,)
    err_arr = input - future

    err = np.max(np.abs(err_arr))
    forcast_err.append(err)

    #if err>1 and err_count!=None:
    #    err_count+=1

    if plot and err>1.0:
        fig, axs = plt.subplots(3)
        axs[0].plot(np.append(sample_profile[(index-prediction_days):(index)], input))
        axs[1].plot(sample_profile[(index-prediction_days):(index+pred_amount)])
        axs[2].plot(np.abs(err_arr))
        plt.show()

    if err_count==None:
        return forcast_err
    else:
        return forcast_err, err_count


def test_model_estimate(scaler, index, model, sample_profile, prediction_days, plot, forcast_err, err_count=None, pred_amount=25):
    mean_val = 0.27201455577239575
    var_val = 0.08216779182643608

    input = scaler.transform(sample_profile[index-prediction_days:index].reshape(-1,1))
    future = sample_profile[index:index+pred_amount]
    #input = sample_profile[index-prediction_days:index].reshape(-1,1)

    lst = []
    lst.append(input.reshape(prediction_days,1))
    lst = np.array(lst)
    pred = model.predict(lst)
    #pred = pred[0]
    input = pred

    #input = input.reshape(-1,1)
    input = scaler.inverse_transform(input.reshape(-1,1))

    input = input.reshape(pred_amount,)
    err_arr = input - future

    err = np.abs(err_arr)
    err_mean = np.mean(err)

    err = (err_mean-mean_val)**2/var_val
    print(err)

    forcast_err.append(err)

    #if err>1 and err_count!=None:
    #    err_count+=1

    if plot and err>1.0:
        fig, axs = plt.subplots(3)
        axs[0].plot(np.append(sample_profile[(index-prediction_days):(index)], input))
        axs[1].plot(sample_profile[(index-prediction_days):(index+pred_amount)])
        axs[2].plot(np.abs(err_arr))
        plt.show()

    if err_count==None:
        return forcast_err
    else:
        return forcast_err, err_count


def make_profile(val1, val2, val3, delay_num):

    y1 = create_exp_curve(y_start=50, y_stop=val1, n_points=100, k=7)
    y2 = create_exp_curve(y_start=val1, y_stop=55, n_points=250, k=3)

    y_change = np.append(y1, np.zeros(shape=(50,))+val1)
    y = np.append(y_change, y2)

    y1 = create_exp_curve(y_start=55, y_stop=val2, n_points=100, k=7)
    y2 = create_exp_curve(y_start=val2, y_stop=60, n_points=300, k=3)

    y_change = np.append(y1, np.zeros(shape=(25,))+val2)
    y_change = np.append(y_change, y2)
    y = np.append(y, y_change)

    y1 = create_exp_curve(y_start=60, y_stop=val3, n_points=100, k=7)
    y2 = create_exp_curve(y_start=val3, y_stop=50, n_points=450, k=4)

    y_change = np.append(y1, np.zeros(shape=(100,))+val3+0.05*np.random.normal(loc=0.0, scale=1.0, size=100))
    y_change = np.append(y_change, y2)
    y = np.append(y, y_change)

    profile = np.copy(y)


    b = signal.firwin(100, 0.05)

    profile = signal.filtfilt(b,1,profile)

    sample_profile = np.zeros(shape=(delay_num,)) + 50 + 0.05*np.random.normal(loc=0.0, scale=1.0, size=delay_num)
    for i in range(1):
        interm = profile+0.05*np.random.normal(loc=0.0, scale=1.0, size=profile.shape[0])
        sample_profile = np.append(sample_profile, interm)

    return sample_profile




#plt.plot(profile)
#plt.show()

full_profile = np.zeros(shape=(1))+50
for i in range(40):
    rand_size = random.randint(100,400)
    interm = np.append(np.zeros(shape=(rand_size))+50+0.05*np.random.normal(loc=0.0, scale=1.0, size=rand_size),make_profile(60, 70, 65, 0)) 
    full_profile = np.append(full_profile, interm)


#plt.plot(full_profile)
#plt.show()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(full_profile.reshape(-1,1))
#scaled_data = full_profile.reshape(-1,1)

prediction_days = 200
future_days = 25

# Initialize empty lists for training data input and output
x_train = []
y_train = []

# Iterate through the scaled data, starting from the prediction_days index
for x in range(prediction_days, len(scaled_data)-future_days):
    # Append the previous 'prediction_days' values to x_train
    x_train.append(scaled_data[x - prediction_days:x, :])
    # Append the current value to y_train
    y_train.append(scaled_data[x:x+future_days, :])

# Convert the x_train and y_train lists to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, shuffle=False)

import keras
from keras.api.saving import load_model
from keras import losses
from keras import optimizers

model= keras.Sequential()
model.add(keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu'))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
#model.add(keras.layers.Dropout(0.01))
model.add(keras.layers.Dense(future_days))


'''
model.summary()
model.compile(loss='mse',optimizer=optimizers.Adam(learning_rate=1E-3))
model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=15, batch_size = 16)#32)
model.save('lstm_forcast_step2.keras')
'''
model = load_model('lstm_forcast_step2.keras')


#'''
plt_errs = True
err_lst = []

sample_profile = make_profile(59, 70, 65, 150)
for i in range(8):
    err_lst = test_model_estimate(scaler, i*25+200, model, sample_profile, prediction_days, plt_errs, err_lst)

sample_profile = make_profile(61, 70, 65, 150)
for i in range(8):
    err_lst = test_model_estimate(scaler, i*25+200, model, sample_profile, prediction_days, plt_errs, err_lst)

sample_profile = make_profile(60, 71, 65, 150)
for i in range(8):
    err_lst = test_model_estimate(scaler, i*25+550, model, sample_profile, prediction_days, plt_errs, err_lst)

sample_profile = make_profile(60, 69, 65, 150)
for i in range(8):
    err_lst = test_model_estimate(scaler, i*25+550, model, sample_profile, prediction_days, plt_errs, err_lst)

sample_profile = make_profile(60, 70, 64, 150)
for i in range(8):
    err_lst = test_model_estimate(scaler, i*25+950, model, sample_profile, prediction_days, plt_errs, err_lst)

sample_profile = make_profile(60, 70, 66, 150)
for i in range(8):
    err_lst = test_model_estimate(scaler, i*25+950, model, sample_profile, prediction_days, plt_errs, err_lst)

plt.plot(err_lst)
plt.show()
#'''

cycle_count = 0
err_count = 0
err_arr = np.zeros(shape=1)
for i in range(20):
    err_lst = []
    sample_profile = make_profile(60, 70, 65, random.randint(100, 200))
    rand_size = random.randint(100,200)
    sample_profile = np.append(sample_profile, np.zeros(shape=(rand_size))+50+0.05*np.random.normal(loc=0.0, scale=1.0, size=rand_size))
    for i in range(int((len(sample_profile)-200)/25)):
        if i!=0:
            err_lst, err_count = test_model_estimate(scaler, int(25*i+200), model, sample_profile, prediction_days, True, err_lst, err_count)
            cycle_count+=1
            #print('Errs: ' + str(err_count) + ' Cycles: ' + str(cycle_count))
            err_arr_app = np.array(err_lst)
            err_arr = np.append(err_arr, err_arr_app)


print(np.mean(err_arr))
print(np.var(err_arr))


plt.hist(err_arr)
plt.show()


for e in err_lst:
    print(e)
