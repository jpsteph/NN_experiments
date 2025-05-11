
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.preprocessing import StandardScaler
from scipy import signal
import random

from sklearn.model_selection import train_test_split



class Scale():
    def __init__(self, arr):
        self.mean_val = np.mean(arr)
        self.std = np.std(arr)

    def transform(self, arr):
        return (arr - self.mean_val)/(self.std)

    def inv_transform(self, arr):
        return (arr*self.std) + self.mean_val



def create_exp_curve(y_start, y_stop, n_points, k):

    # Create x array (normalized from 0 to 1)
    x = np.linspace(0, 1, n_points)

    # Exponential rise
    y = y_start + (y_stop - y_start) * (1 - np.exp(-k * x))
    return y


def test_model(scaler, index, model, sample_profile, prediction_days, plot, forcast_err, err_count=None, pred_amount=25):

    input = scaler.transform(sample_profile[index-prediction_days:index])
    future = sample_profile[index:index+pred_amount]
    #input = sample_profile[index-prediction_days:index].reshape(-1,1)

    lst = []
    lst.append(input.reshape(prediction_days,1))
    lst = np.array(lst)
    pred = model.predict(lst)
    #pred = pred[0]
    input = pred

    #input = input.reshape(-1,1)
    input = scaler.inv_transform(input)

    input = input.reshape(pred_amount,)
    err_arr = input - future

    err = np.abs(err_arr)
    forcast_err.append(err)

    #if err>1 and err_count!=None:
    #    err_count+=1

    if plot:
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
    
    err_lst = []
    index_amount = 200
    for i in range(96):
        sample_profile_ind = sample_profile[12*i:i*12+index_amount]
        input = scaler.transform(sample_profile_ind).reshape(1, index_amount, 1)

        pred_inv = model.predict(input)
        
        #input = input.reshape(-1,1)
        pred = scaler.inv_transform(pred_inv[0,:,0])

        err_arr = np.abs(pred - sample_profile_ind)
        err_est = np.mean(err_arr)
        err_lst.append(err_est)
        print(err_est)

        #plt.plot(sample_profile_ind)
        #plt.plot(pred)
        #plt.show()

    #plt.plot(err_lst)
    #plt.show()

    forcast_err.append(err_est)

    #if err>1 and err_count!=None:
    #    err_count+=1

    if plot and np.max(err_lst)>0.5:
        fig, axs = plt.subplots(3)
        axs[0].plot(err_lst)
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


    sample_profile = np.zeros(shape=(delay_num,)) + 50 + 0.05*np.random.normal(loc=0.0, scale=1.0, size=delay_num)
    for i in range(1):
        interm = profile+0.05*np.random.normal(loc=0.0, scale=1.0, size=profile.shape[0])
        sample_profile = np.append(sample_profile, interm)

    return sample_profile



full_profile = np.zeros(shape=(1))+50
for i in range(40):
    rand_size = random.randint(100,400)
    interm = np.append(np.zeros(shape=(rand_size))+50+0.05*np.random.normal(loc=0.0, scale=1.0, size=rand_size),make_profile(60, 70, 65, 0)) 
    full_profile = np.append(full_profile, interm)


#plt.plot(full_profile)
#plt.show()

#scaler = StandardScaler()
#scaled_data = scaler.fit_transform(full_profile.reshape(-1,1))
#scaled_data = full_profile.reshape(-1,1)

scaler = Scale(full_profile)

prof = make_profile(60, 70, 65, 0)
prof_t = scaler.transform(prof)
prof_inv = scaler.inv_transform(prof_t)

# Initialize empty lists for training data input and output
x_train = []
y_train = []

# Iterate through the scaled data, starting from the prediction_days index
for x in range(400):
    prof = make_profile(60, 70, 65, random.randint(100,300))
    
    x_input = scaler.transform(prof.reshape(-1,1))
    y_input = scaler.transform(prof.reshape(-1,1))
    for i in range(1300):
        x_train.append(x_input[i:i+200])
        y_train.append(y_input[i:i+200])

# Convert the x_train and y_train lists to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, shuffle=False)


# Initialize empty lists for training data input and output
x_train_tamper = []
y_train_tamper = []

# Iterate through the scaled data, starting from the prediction_days index
for x in range(400):
    offset = random.uniform(-2, 2) 
    offset2 = random.uniform(-2, 2) 
    offset3 = random.uniform(-2, 2) 

    prof_offset = make_profile(60+offset, 70+offset2, 65+offset3, random.randint(100,300))

    x_input = scaler.transform(prof_offset.reshape(-1,1))
    y_input = scaler.transform(prof_offset.reshape(-1,1))
    for i in range(int(1300)):
        x_train_tamper.append(x_input[i:i+200])
        y_train_tamper.append(y_input[i:i+200])

x_train_tamper = np.array(x_train_tamper)
y_train_tamper = np.array(y_train_tamper)

x_train_tamper, x_test_tamper, y_train_tamper, y_test_tamper = train_test_split(x_train_tamper, y_train_tamper, test_size = 0.2, shuffle=False)


import keras
from keras.api.saving import load_model
from keras import losses
from keras import optimizers
from keras.api.callbacks import Callback
from keras.api.callbacks import ModelCheckpoint
from keras.api.losses import Huber

from keras.api.callbacks import LearningRateScheduler
import math

model_name = 'rnn_auto_3.keras' #'cnn_exp.keras'

# Exponential increase function
def exponential_lr(epoch, lr):
    k = 0.05  # Growth rate (tune as needed)
    return lr * math.exp(k)

lr_scheduler = LearningRateScheduler(exponential_lr, verbose=1)

checkpoint_cb = ModelCheckpoint(
    model_name,           # File to save the model
    save_best_only=True,       # Only save when val_loss improves
    monitor='val_loss',        # What to monitor
    mode='min',                # 'min' because we want to minimize val_loss
    verbose=1                  # Print info when saving
)
class tamper_val_call(Callback):
    def __init__(self, val_data):
        self.x_val_data, self.y_val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        val_loss = self.model.evaluate(self.x_val_data, self.y_val_data, verbose=0)

        print(f'\nEpoch {epoch+1} - Tampered Val loss: {val_loss:.4f}')
kernel_size = 6

model = keras.Sequential(
    [
        keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        keras.layers.Conv1D(
            filters=32,
            kernel_size=kernel_size,
            padding="same",
            strides=1,
            activation="relu",
        ),
        keras.layers.Conv1D(
            filters=16,
            kernel_size=kernel_size,
            padding="same",
            strides=1,
            activation="relu",
        ),
        #keras.layers.Dropout(rate=0.01),
        keras.layers.Conv1D(
            filters=2,
            kernel_size=kernel_size,
            padding="same",
            strides=1,
            activation="relu",
        ),
        keras.layers.Conv1DTranspose(
            filters=2,
            kernel_size=kernel_size,
            padding="same",
            strides=1,
            activation="relu",
        ),
        #keras.layers.Dropout(rate=0.01),
        keras.layers.Conv1DTranspose(
            filters=16,
            kernel_size=kernel_size,
            padding="same",
            strides=1,
            activation="relu",
        ),
        keras.layers.Conv1DTranspose(
            filters=32,
            kernel_size=kernel_size,
            padding="same",
            strides=1,
            activation="relu",
        ),
        keras.layers.Conv1DTranspose(filters=1, kernel_size=kernel_size, padding="same"),
    ]
)

model = keras.models.Sequential([
    keras.layers.LSTM(30, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    keras.layers.LSTM(10),

    keras.layers.RepeatVector(x_train.shape[1], input_shape=[10]),
    keras.layers.LSTM(30, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(x_train.shape[2]))
])


#'''
model.summary()
#Huber()
model.compile(loss='mse',optimizer=optimizers.Adam(learning_rate=1E-3))
#model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=100, batch_size =128, callbacks=[tamper_val_call((x_test_tamper, y_test_tamper)), checkpoint_cb])#, lr_scheduler])
#model.save(model_name)
#'''
model = load_model(model_name)
model.summary()

#pred = model.predict(x_train[0].reshape(1, 1475, 1))
#plt.plot(pred[0,:,0])
#plt.plot(x_train[0])
#plt.show()

#model.evaluate(x_train, y_train)
#model.evaluate(x_train_tamper, y_train_tamper)

#'''
plt_errs = True
err_lst = []

prediction_days = []

while 1:
    sample_profile = make_profile(60, 70, 65, random.randint(100, 300))
    err_lst = test_model_estimate(scaler, 550, model, sample_profile, prediction_days, plt_errs, err_lst)

sample_profile = make_profile(60, 70, 66, 150)
err_lst = test_model_estimate(scaler, 550, model, sample_profile, prediction_days, plt_errs, err_lst)

sample_profile = make_profile(58, 70, 65, 150)
err_lst = test_model_estimate(scaler, 200, model, sample_profile, prediction_days, plt_errs, err_lst)

sample_profile = make_profile(60, 71, 65, 0)
err_lst = test_model_estimate(scaler, 550, model, sample_profile, prediction_days, plt_errs, err_lst)


'''
sample_profile = make_profile(60, 70, 64, 150)
for i in range(8):
    err_lst = test_model_estimate(scaler, i*25+950, model, sample_profile, prediction_days, plt_errs, err_lst)

sample_profile = make_profile(60, 70, 66, 150)
for i in range(8):
    err_lst = test_model_estimate(scaler, i*25+950, model, sample_profile, prediction_days, plt_errs, err_lst)

plt.plot(err_lst)
plt.show()
'''
#'''

cycle_count = 0
err_count = 0
err_arr = np.zeros(shape=1)
#while(1):
for i in range(200):
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
            print(cycle_count)

print(np.mean(err_arr))
print(np.var(err_arr))


plt.hist(err_arr)
plt.show()


for e in err_lst:
    print(e)
