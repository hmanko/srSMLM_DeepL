# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:55:28 2020
@author: hmanko
"""
'''
### https://github.com/hmanko/NN_sr
'''

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


image_width = 64
image_height = 16
image_chanels = 1

#############################################
###     TO DOWNLOAD IMAGES FROM STACK     ###
#############################################
X_train_25_2 = imread('D:/My Library_2/U-net/For_Network/x_new/25_2_patch_tr.tif')
X_train_50_1 = imread('D:/My Library_2/U-net/For_Network/x_new/50_1_patch_tr.tif')
X_train_50_2 = imread('D:/My Library_2/U-net/For_Network/x_new/50_2_patch_tr.tif')
X_train_75_1 = imread('D:/My Library_2/U-net/For_Network/x_new/75_1_patch_tr.tif')
X_train_75_2 = imread('D:/My Library_2/U-net/For_Network/x_new/75_2_patch_tr.tif')
X_train_100_1 = imread('D:/My Library_2/U-net/For_Network/x_new/100_1_patch_tr.tif')
X_train_100_2 = imread('D:/My Library_2/U-net/For_Network/x_new/100_2_patch_tr.tif')
X_train_125_2 = imread('D:/My Library_2/U-net/For_Network/x_new/125_2_patch_tr.tif')
X_train_125_3 = imread('D:/My Library_2/U-net/For_Network/x_new/125_3_patch_tr.tif')
X_train_150_1 = imread('D:/My Library_2/U-net/For_Network/x_new/150_1_patch_tr.tif')
X_train_150_2 = imread('D:/My Library_2/U-net/For_Network/x_new/150_2_patch_tr.tif')
X_noise = imread('D:/My Library_2/U-net/For_Network/xx/Noise.tif')
X_noise2 = imread('D:/My Library_2/U-net/For_Network/xx/Noise2.tif')
X_noise3 = imread('D:/My Library_2/U-net/For_Network/xx/Noise3.tif')

X_train = np.concatenate((X_noise/1.5, X_train_25_2,X_train_50_1,X_train_50_2,
                          X_train_75_1,X_train_75_2,X_train_100_1,
                          X_train_125_2,X_noise2/1.5,X_train_125_3,X_train_150_1,X_train_150_2,X_noise3/1.5 ), axis=0)

del X_train_25_2,X_train_50_1,X_train_50_2
del X_train_75_1,X_train_75_2,X_train_100_1,X_train_100_2
del X_train_125_2,X_train_125_3,X_train_150_1,X_train_150_2, X_noise, X_noise2, X_noise3

X_train = X_train/60000


X_train = np.delete(X_train, [355214], axis=0)
X_train = X_train.reshape((-1, image_height, image_width, 1))
###//////////////
Y_train_25_2 = imread('D:/My Library_2/U-net/For_Network/y_new/25_2_patch_sum.tif')
Y_train_50_1 = imread('D:/My Library_2/U-net/For_Network/y_new/50_1_patch_sum.tif')
Y_train_50_2 = imread('D:/My Library_2/U-net/For_Network/y_new/50_2_patch_sum.tif')
Y_train_75_1 = imread('D:/My Library_2/U-net/For_Network/y_new/75_1_patch_sum.tif')
Y_train_75_2 = imread('D:/My Library_2/U-net/For_Network/y_new/75_2_patch_sum.tif')
Y_train_100_1 = imread('D:/My Library_2/U-net/For_Network/y_new/100_1_patch_sum.tif')
Y_train_100_2 = imread('D:/My Library_2/U-net/For_Network/y_new/100_2_patch_sum.tif')
Y_train_125_2 = imread('D:/My Library_2/U-net/For_Network/y_new/125_2_patch_sum.tif')
Y_train_125_3 = imread('D:/My Library_2/U-net/For_Network/y_new/125_3_patch_sum.tif')
Y_train_150_1 = imread('D:/My Library_2/U-net/For_Network/y_new/150_1_patch_sum.tif')
Y_train_150_2 = imread('D:/My Library_2/U-net/For_Network/y_new/150_2_patch_sum.tif')
Y_noise = imread('D:/My Library_2/U-net/For_Network/yy/Noise.tif')
Y_noise2 = imread('D:/My Library_2/U-net/For_Network/yy/Noise2.tif')
Y_noise3 = imread('D:/My Library_2/U-net/For_Network/yy/Noise3.tif')

Y_train = np.concatenate((Y_noise/1.5, Y_train_25_2, Y_train_50_1,Y_train_50_2,
                          Y_train_75_1,Y_train_75_2,Y_train_100_1,
                          Y_train_125_2,Y_noise2/1.5,Y_train_125_3,Y_train_150_1, Y_train_150_2, Y_noise3/1.5), axis=0)

del Y_train_25_2,Y_train_50_1,Y_train_50_2
del Y_train_75_1,Y_train_75_2,Y_train_100_1,Y_train_100_2
del Y_train_125_2,Y_train_125_3,Y_train_150_1,Y_train_150_2, Y_noise, Y_noise2, Y_noise3

Y_train = Y_train/Y_train.max()
Y_train = Y_train.reshape(-1, image_height, image_width, 1)

#############################################
# Saving
imsave('C:/Users/hmanko/Desktop/X_train.tif', X_train)
imsave('C:/Users/hmanko/Desktop/Y_train.tif', Y_train)
#############################################


######################################
######### Image augmentation 
#  It is required to expand a bit spectral part to make the final trained model be able to work properly with other type of data
for i in range(0, len(X_train)):
    datagen = ImageDataGenerator(zoom_range=[0.7,1.0])
    it_x = datagen.flow(expand_dims(X_train[i, :, 16:], 0), batch_size=1)
    X_train[i] = np.concatenate((X_train[i, :,:16],it_x[0].reshape( 16, 48,1)), axis=1)
    it_y = datagen.flow(expand_dims(Y_train[i, :, 16:], 0), batch_size=1)
    Y_train[i] = np.concatenate((Y_train[i, :,:16],it_y[0].reshape( 16, 48,1)), axis=1)

#############################################
#
#    /\    /\    ___   ___|  __  |
#   /  \  /  \  |   | |   | |__| |
#  /    \/    \ |___| |___| |__  |___
#
#############################################

image_width = 64
image_height = 16
image_chanels = 1


X_train =  imread('C:/Users/hmanko/Desktop/X_train.tif')
Y_train = imread('C:/Users/hmanko/Desktop/Y_train.tif')

inputs = Input((image_height, image_width, image_chanels))

c0=Conv2D(16, (6,6),activation = 'elu',kernel_initializer='he_normal', padding = 'same')(inputs)
c0=BatchNormalization(axis=-1)(c0)
c0=LeakyReLU(alpha=0.2)(c0)
c0=Conv2D(16, (6,6),activation = 'elu',strides = 2,kernel_initializer='he_normal', padding = 'same')(c0)
c0=BatchNormalization(axis=-1)(c0)
c0=LeakyReLU(alpha=0.2)(c0)

c1=Conv2D(32, (6,6),activation = 'elu',kernel_initializer='he_normal', padding = 'same')(c0)
c1=BatchNormalization(axis=-1)(c1)
c1=LeakyReLU(alpha=0.2)(c1)
c1=Conv2D(32, (6,6),activation = 'elu',strides = 2, kernel_initializer='he_normal', padding = 'same')(c1)
c1=BatchNormalization(axis=-1)(c1)
c1=LeakyReLU(alpha=0.2)(c1)

c2=Conv2D(64, (6,6),activation = 'elu',kernel_initializer='he_normal', padding = 'same')(c1)
c2=BatchNormalization(axis=-1)(c2)
c2=LeakyReLU(alpha=0.2)(c2)
c2=Conv2D(64, (6,6),activation = 'elu',strides = 2,kernel_initializer='he_normal',padding = 'same')(c2)
c2=BatchNormalization(axis=-1)(c2)
c2=LeakyReLU(alpha=0.2)(c2)

c3=Conv2D(128, (6,6),activation = 'elu',kernel_initializer='he_normal', padding = 'same')(c2)
c3=BatchNormalization(axis=-1)(c3)
c3=LeakyReLU(alpha=0.3)(c3)

u4=Conv2DTranspose(64, (6,6), padding ='same')(c3)
u4=concatenate([u4,c2])
c4=UpSampling2D(size=2)(u4)
c4=Conv2D(64, (6,6),activation = 'elu',kernel_initializer='he_normal', padding = 'same')(c4)
c4=BatchNormalization(axis=-1)(c4)
c4=LeakyReLU(alpha=0.2)(c4)

u5=Conv2DTranspose(32, (6,6), padding ='same')(c4)
u5=concatenate([u5,c1])
c5=UpSampling2D(size=2)(u5)
c5=Conv2D(32, (6,6),activation = 'elu',kernel_initializer='he_normal', padding = 'same')(c5)
c5=BatchNormalization(axis=-1)(c5)
c5=LeakyReLU(alpha=0.2)(c5)

u6=Conv2DTranspose(16, (6,6), padding ='same')(c5)
u6=concatenate([u6,c0])
c6=UpSampling2D(size=2)(u6)
c6=Conv2D(16, (6,6),activation = 'elu',kernel_initializer='he_normal', padding = 'same')(c6)
c6=BatchNormalization(axis=-1)(c6)
c6=LeakyReLU(alpha=0.2)(c6)

outputs = Conv2D(1, (1,1), activation ='relu')(c6)

model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
model.compile(optimizer = 'rmsprop', loss="mean_squared_error")
model.summary()
          
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=4, min_lr=0.001)
earlyStop = keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)
callbacks_list = [earlyStop, reduce_lr]

history = model.fit(X_train,Y_train,validation_split=0.1, batch_size=80, epochs=40)

model.save("D:/My Library_2/model_mix_30.07_19_BS_50(6,6)_.h5")

#############################################
##    Downloading the test patches
X_test = imread('/path/*.tif')
X_test = X_test/60000
X_test = X_test.reshape(-1, image_height, image_width, 1) 

Y_test = imread('/path/*.tif')
Y_test = Y_test/60000
Y_test = Y_test.reshape(-1, image_height, image_width, 1) 

prediction = model.predict(X_test)   

###################### 
num = random.randint(1,len(X_test))
plt.figure(figsize = (25,10))
plt.subplot(221)
plt.plot(list(range(0,64)),X_test[num,10,:,0])
plt.plot(list(range(0,64)),Y_test[num, 8,:,0])
plt.plot(list(range(0,64)),prediction[num, 10,:,0])
plt.legend(['X_pos', 'Y_pos','prediction'])
plt.subplot(222)
plt.imshow(prediction[num,:,:,0])
plt.title("Prediction")
plt.subplot(223)
plt.imshow(X_test[num,:,:,0])
plt.title("Test example")
plt.subplot(224)
plt.imshow(Y_test[num,:,:,0])
plt.title("Clean image")
plt.show()

imsave('D:/1/Data/GataQuand__15-150mW/50_4_pred.tif', prediction)

##  To download already trained model
model = load_model("D:/My Library_2/model_mix_19.08_25_BS_80(6,6)_.h5")


