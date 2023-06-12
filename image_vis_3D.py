#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 18:00:09 2022

@author: Xiao

To visualize 3D images

"""
import sys
sys.path.insert(1,'/home/xiao/Inverse_MS_Design/Python/Common_lib/')

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def to_rgb(rod):
    rgb = (rod+np.sqrt(2)-1)/(2*(np.sqrt(2)-1))
    return rgb

def plot_3D(images,image_num=0,image_format='svg',save_dir=''):
    #data_file = '/Users/Xiao/cGAN_Project/ML/CNN/dataset/20220706/preprocessed/25/input.npy'   
    tesrsize = 32
    #image_num = 0 # which image to visualize
    # Controll Tranperency
    alpha = 1.0
    
    # Change the Size of Graph using
    # Figsize
    #fig = plt.figure(figsize=(10, 10))
     
    # Generating a 3D sine wave
    ax = plt.axes(projection='3d')
    
    image = np.ones([tesrsize,tesrsize,tesrsize,4])
    image_bi = np.ones([tesrsize,tesrsize,tesrsize,4])
    
    for z in range(0,tesrsize):
        for y in range(0,tesrsize):
            for x in range(0,tesrsize):
                #rgb = to_rgb(images[image_num][x][y][z][2:5])
                rgb = to_rgb(images[0][x][y][z][1:4])
                image[x][y][z] = np.append(rgb,alpha)
                if images[0][x][y][z][1]==0:
                    image_bi[x][y][z][0] = 0
                else:
                    image_bi[x][y][z][1] = 0
                    
    image[image>1]=1
    image[image<0]=0
    
    data = np.ones([tesrsize,tesrsize,tesrsize])
    
    # trun off/on axis
    ax.axis('off')
#    ax.axis('on')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
     
    # Voxels is used to customizations of
    # the sizes, positions and colors.
    
    # match stress/strain
    image = np.flip(image,(0,1)) #matches neper vis
#    lightsource = matplotlib.colors.LightSource(0,240)
#    ax.view_init(elev =30., azim= 120.)
    
    #match neper
    ax.view_init(elev = 30, azim= 120)    #matches neper vis
    lightsource = matplotlib.colors.LightSource(0,150)
    ax.voxels(data, facecolors = image, edgecolors = image, lightsource=lightsource)

    if save_dir != '':
       plt.savefig(save_dir+"_{0}.{1}".format(image_num,image_format))
    return 0

# plot results 3D image in RGB color
def plot_3D_res(images,image_num=0,cb='False',image_format='png',stress_min='',stress_max='',save_dir='',axis='off'):
    image = images[0]
    if type(image)!=np.ndarray:
        image=image.numpy()
    #data_file = '/Users/Xiao/cGAN_Project/ML/CNN/dataset/20220706/preprocessed/25/input.npy'   
    tesrsize = 32
    ax = plt.axes(projection='3d')
    image = np.flip(image,(0,1)) #matches neper vis
    image =  image.reshape((32,32,32))
    cmap = plt.get_cmap("jet")
    # set the range of the colormap
    if stress_max=='':
        if stress_min=='':
            norm= plt.Normalize(image.min(), image.max())
        else:
            norm= plt.Normalize(stress_min, image.max())
    else:
        if stress_min=='':
            norm= plt.Normalize(image.min(), stress_max())
        else:
            norm= plt.Normalize(stress_min, stress_max)
        
    data = np.ones([tesrsize,tesrsize,tesrsize])
    
    # trun off/on axis
    ax.axis(axis)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
     
    # Voxels is used to customizations of
    # the sizes, positions and colors.
#    ax.view_init(elev =30., azim= 120.)
#    lightsource = matplotlib.colors.LightSource(0,240)
    
    #match neper
    ax.view_init(elev = 30, azim= 120)    #matches neper vis
    lightsource = matplotlib.colors.LightSource(0,60)
    
    ax.voxels(data, facecolors = cmap(norm(image)), edgecolors = cmap(norm(image)), lightsource=lightsource)

    
    m = matplotlib.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    if cb == True:
        plt.colorbar(m,fraction=0.0275, pad=0.125)
    if save_dir != '':
        plt.savefig(save_dir+"_{0}.{1}".format(image_num,image_format))
        
    plt.figure().clear()
    return image
