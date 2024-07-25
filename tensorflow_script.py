import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import fashion_mnist, cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Flatten, Dropout
from tensorflow.keras.utils import to_categorical as tc
import numpy as np
import matplotlib.pyplot as plt

###set parameters###
num_models = 8
epochs_per_round = 5
rounds = 10
bs = 128

###load data###
(x,y),(x_test,y_test) = cifar10.load_data()
x  = x/255.0
y = tc(y)
x_test = x_test/255.0
y_test = tc(y_test)

###you can probably use your function here###
def federate_data(x,y):
    ###???###
    #return(x,y)
    pass

def split_data(x, y, n_splits, equal_sizes):
    if equal_sizes:
        split_sizes = [len(x) // n_splits for _ in range(n_splits)]
    else:
        total_size = len(x)
        split_sizes = []
        for i in range(n_splits - 1):
            split = random.randrange(1, total_size)
            split_sizes.append(split)
            total_size -= split
        split_sizes.append(total_size)
    x_splits = []
    y_splits = []
    start_idx = 0
    for size in split_sizes:
        end_idx = start_idx + size
        subset_x = x[start_idx:end_idx]
        subset_y = y[start_idx:end_idx]
        x_splits.append(subset_x)
        y_splits.append(subset_y)
        start_idx = end_idx
    return x_splits, y_splits, split_sizes

###define model###
def get_model(classes=10,input_shape=(32,32,3)):
    model = Sequential()
    model.add(Conv2D(32,3,1,padding='same',activation='relu',input_shape=input_shape))
    model.add(Dropout(.1))
    model.add(Conv2D(32,3,1,padding='same',activation='relu'))
    model.add(Dropout(.1))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization())
    model.add(Conv2D(64,3,1,padding='same',activation='relu'))
    model.add(Dropout(.1))
    model.add(Conv2D(64,3,1,padding='same',activation='relu'))
    model.add(Dropout(.1))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization())
    model.add(Conv2D(128,3,1,padding='same',activation='relu'))
    model.add(Dropout(.1))
    model.add(Conv2D(128,3,1,padding='same',activation='relu'))
    model.add(Dropout(.1))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(classes,activation='softmax'))
    model.compile(optimizer='nadam',loss='categorical_crossentropy',metrics=['acc'])
    #model.summary()
    return(model)

###federated learning###
global_model = get_model()
weights_list = [global_model.get_weights()]*num_models

for r in range(rounds):
    print('Starting round '+str(r+1))
    for m in range(num_models):
        print('training model ' + str(m+1))
        global_model.set_weights(weights_list[m])
        global_model.fit(x[:int(len(x)/num_models)],y[:int(len(x)/num_models)],epochs=epochs_per_round,batch_size=bs,validation_data=(x_test,y_test),verbose=0)
        weights_list[m]=global_model.get_weights()
    global_weights=np.mean(np.array(weights_list,dtype='object'),axis=0)
    weights_list = [global_weights]*num_models
    global_model.set_weights(global_weights)
    print('global model performance after '+str(r+1)+' round:')
    global_model.evaluate(x_test,y_test,batch_size=1024)