#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 04:08:32 2021

@author: abdullah
"""

import os
from PIL import Image
from PIL import ImageFilter
import numpy as np
import keras
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D
from keras.models import Sequential

os.environ['CUDA_VISIBLE_DEVICES'] = ''

num_classes = 10
epochs = 50
batch_size = 4

img_rows, img_cols = 32, 32

os.chdir("flower/flower_images")

df = pd.read_csv('flower_labels.csv')#load images' names and labels
names = df['file'].values
labels = df['label'].values.reshape(-1,1)

data = []



for name in names:
    img = Image.open(name)#shape is 128x128x4
    img = img.resize((img_rows, img_cols))#resize image into fixed size 32x32x4
    img = np.array(img)[np.newaxis, :, :, :3]#add new axis and new size is 1x32x32x3
    data.append(img)
    
# =============================================================================
data = np.concatenate(data)#concatenate images, shape is 209x32x32x3
 
x_train = data[:177].astype(np.float32)
y_train = labels[:177]
x_test = data[177:].astype(np.float32)# set the last 32 images as test dataset
y_test = labels[177:]
x_train /= 255  # Normalization that scales the pixel values to a range between 0 and 1
x_test /= 255  # Normalization that scales the pixel values to a range between 0 and 1
# =============================================================================

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)#convert label into one-hot vector

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid', input_shape=[32,32,3]))#Convolution
model.add(Activation('relu'))#Activation function
model.add(AveragePooling2D(pool_size=(30, 30)))#30x30 average pooling
model.add(Flatten())# shape equals to [batch_size, 32] 32 is the number of filters
model.add(Dense(10))#Fully connected layer
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.01, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


def train():
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
    
train()

