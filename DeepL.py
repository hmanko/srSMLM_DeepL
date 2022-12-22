#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 10:28:05 2022

@author: hannamanko
"""

import keras
import tensorflow as tf
import os

import numpy as np
from matplotlib import pyplot as plt
import random
from tifffile import imread, imsave
from PIL import Image


from skimage import io, img_as_ubyte
from skimage.transform import resize, rescale
import random
import pandas as pd
from tqdm import tqdm 

from tensorflow.keras.layers import Lambda,Input,Conv2D,BatchNormalization,AveragePooling2D,LeakyReLU,Conv2DTranspose,concatenate,UpSampling2D,Dropout
from skimage.morphology import disk, white_tophat

from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.model_selection import train_test_split

image_width = 64
image_height = 16
image_chanels = 1





Xtest = np.load('.../data/Xtest.npz')
Xtest.files
X_test = Xtest['Xtest']

Xtrain = np.load('.../data/Xtrain.npz')
Xtrain.files
X_train = Xtrain['Xtrain']


Ytest = np.load('.../data/Ytest.npz')
Ytest.files
Y_test = Ytest['Ytest']

Ytrain = np.load('.../data/Ytrain.npz')
Ytrain.files
Y_train = Ytrain['Ytrain']


### If required the training set need to be reshaped: 
#X_train = X_train.reshape(len(X_train),image_height,image_width, 1)
#Y_train = Y_train.reshape(len(Y_train),image_height,image_width, 1)


#############################################
#
#    /\    /\    ___   ___|  __  |
#   /  \  /  \  |   | |   | |__| |
#  /    \/    \ |___| |___| |__  |___
#
#############################################
inputs = Input((image_height, image_width, image_chanels))

c0=Conv2D(16, (6,6),activation = LeakyReLU(alpha=0.2),kernel_initializer='he_normal', padding = 'same')(inputs)
c0=BatchNormalization(axis=-1)(c0)
c0=Conv2D(16, (6,6),activation = LeakyReLU(alpha=0.2),strides = 2,kernel_initializer='he_normal', padding = 'same')(c0)
c0=BatchNormalization(axis=-1)(c0)

c1=Conv2D(32, (6,6),activation = LeakyReLU(alpha=0.2),kernel_initializer='he_normal', padding = 'same')(c0)
c1=BatchNormalization(axis=-1)(c1)
c1=Conv2D(32, (6,6),activation = LeakyReLU(alpha=0.2),strides = 2, kernel_initializer='he_normal', padding = 'same')(c1)
c1=BatchNormalization(axis=-1)(c1)

c2=Conv2D(64, (6,6),activation = LeakyReLU(alpha=0.2),kernel_initializer='he_normal', padding = 'same')(c1)
c2=BatchNormalization(axis=-1)(c2)
c2=Conv2D(64, (6,6),activation = LeakyReLU(alpha=0.2),strides = 2,kernel_initializer='he_normal',padding = 'same')(c2)
c2=BatchNormalization(axis=-1)(c2)

c3=Conv2D(128, (6,6),activation = LeakyReLU(alpha=0.2),kernel_initializer='he_normal', padding = 'same')(c2)
c3=BatchNormalization(axis=-1)(c3)

u4=Conv2DTranspose(64, (6,6), padding ='same')(c3)
u4=concatenate([u4,c2])
c4=UpSampling2D(size=2)(u4)
c4=Conv2D(64, (6,6),activation = LeakyReLU(alpha=0.2),kernel_initializer='he_normal', padding = 'same')(c4)
c4=BatchNormalization(axis=-1)(c4)

u5=Conv2DTranspose(32, (6,6), padding ='same')(c4)
u5=concatenate([u5,c1])
c5=UpSampling2D(size=2)(u5)
c5=Conv2D(32, (6,6),activation = LeakyReLU(alpha=0.2),kernel_initializer='he_normal', padding = 'same')(c5)
c5=BatchNormalization(axis=-1)(c5)

u6=Conv2DTranspose(16, (6,6), padding ='same')(c5)
u6=concatenate([u6,c0])
c6=UpSampling2D(size=2)(u6)
c6=Conv2D(16, (6,6),activation = LeakyReLU(alpha=0.2),kernel_initializer='he_normal', padding = 'same')(c6)
c6=BatchNormalization(axis=-1)(c6)

outputs = Conv2D(1, (1,1), activation ='sigmoid')(c6)

model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
model.compile(optimizer = 'adam', loss="mean_squared_error")
model.summary()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=4, min_lr=0.0001)
earlyStop = keras.callbacks.EarlyStopping(patience=4, verbose=1, restore_best_weights=True)
callbacks_list = [earlyStop, reduce_lr]

history = model.fit(X_train,Y_train,validation_split=0.1, batch_size=64, epochs=20)

model.save("path")

