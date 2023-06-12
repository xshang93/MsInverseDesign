#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 10:08:58 2022

@author: Xiao

Model builder for 3D CNN

"""

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import ModelCheckpoint
# from matplotlib import pyplot as plt
# from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv3D, MaxPool3D, GlobalMaxPool3D, GlobalAveragePooling3D, Dropout
from tensorflow.keras.optimizers import SGD,Adam

# import pandas as pd
# import glob
# import numpy as np
# from sklearn.metrics import r2_score
# from joblib import dump, load

# define a function to create the convolutional sections of the model
def conv3D_block(inp, filters=64, bn=True, pool=True, dropout_loc=0.4,act_fun = 'LeakyReLU'):
    _ = Conv3D(filters, kernel_size=(3, 3, 3), activation=act_fun, kernel_initializer='he_uniform')(inp)
    if bn:
        _ = BatchNormalization()(_)
    if pool:
        _ = MaxPool3D()(_)
    if dropout_loc > 0:
        _ = Dropout(dropout_loc)(_)
    return _

# function to build model
def model_build(inp_ch=4,nfilters=[32,32,64,128],bn=[True,True,True,True],
                pool=[False,False,False,False],dropout_loc=[0,0,0,0],dropout_glob=0.2,
                act_fun='LeakyReLU',loss='mse',learning_rate=0.001,epsilon =1e-7,
                amsgrad=False,momentum=0.0, opt='Adam'):
    # specify input image sizes
    img_height = 32; img_width = 32; img_depth = 32;

    input_layer = Input(shape=(img_height, img_width, img_depth, inp_ch))
    _ = conv3D_block(input_layer, filters=nfilters[0], bn=bn[0], pool=pool[0],
                     dropout_loc=dropout_loc[0],act_fun = act_fun)
    _ = conv3D_block(_, filters=nfilters[1],bn=bn[1],pool=pool[1],
                     dropout_loc=dropout_loc[1],act_fun = act_fun)
    _ = conv3D_block(_, filters=nfilters[2],bn=bn[2],pool=pool[2],
                     dropout_loc=dropout_loc[2],act_fun = act_fun)
    _ = conv3D_block(_, filters=nfilters[3],bn=bn[3],pool=pool[3],
                     dropout_loc=dropout_loc[3],act_fun = act_fun)
    _ = GlobalAveragePooling3D()(_)
    
    #_ = Dense(units=1024, activation='LeakyReLU')(_)
    _ = Dense(units=64, activation=act_fun)(_)
    if dropout_glob>0:
        _ = Dropout(dropout_glob)(_)
    output = Dense(units=6)(_)
    
    if opt == 'Adam': 
        optimizer = Adam(learning_rate=learning_rate,epsilon=epsilon,amsgrad=amsgrad)
    else:
        optimizer = SGD(learning_rate = learning_rate,momentum=momentum)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['mae'])
    return model