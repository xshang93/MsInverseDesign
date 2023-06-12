#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 15:43:49 2022

@author: Xiao

Data preprocessing module

"""
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pandas as pd
import numpy as np
import random
import shutil
import glob

def train_dataset_gen(save_name,data_dirs,train_ratio,val_ratio,ch_start,ch_end,std = True):

    input_batches = []
    target_batches = []
    
    for i in range(0,len(data_dirs)):
        if data_dirs[i]:
            input_batch = np.load(data_dirs[i] + 'input.npy')
            input_batches.append(input_batch)
            
            target_batch = pd.read_csv(data_dirs[i]+'stress.csv',index_col=0)
            target_batches.append(target_batch)
            
    all_input_ori = np.concatenate(input_batches,axis=0)
    all_target_ori = pd.concat(target_batches)
    
    all_input = all_input_ori[:,:,:,:,ch_start:ch_end]
    all_target = all_target_ori.iloc[:,1:7]/1000000 # convert units to MPa

    
    # ----------------------- loading data finishes -------------------------------------#
    
    trainval_size = int(len(all_input)*train_ratio)
    train_size = int(trainval_size*(1-val_ratio))
    val_size = trainval_size-train_size
#    test_size = len(all_input)-trainval_size
    
    # ------------ sepearte data randomly to train, val and test ----------------#
    
    trainval_idx = random.sample(range(0,len(all_target)),trainval_size)
    trainval_input = all_input[trainval_idx]
    trainval_target = all_target.iloc[trainval_idx]
    
    val_idx = random.sample(range(0,len(trainval_target)),val_size)
    val_input = trainval_input[val_idx]
    val_target = trainval_target.iloc[val_idx]
    
    train_input = np.delete(trainval_input,val_idx,0)
    train_target = trainval_target.drop(index=val_target.index)
    
    if std:
        # ------------ Standardize output data with training samples ----------------#
    
        sc = StandardScaler()
        sc.fit(train_target)
        
        train_target = sc.transform(train_target)
        val_target = sc.transform(val_target)
        #test_target = sc.transform(test_target)
        
        # save the standard scaler for model testing
        dump(sc,'std_scaler'+save_name+'.bin', compress=True)
    
    # save the index for testing
    np.save('train_index_'+save_name,trainval_idx)
    
    return train_input,train_target,val_input,val_target


def test_dataset_gen(save_name,data_dirs,result_dir,ch_start,ch_end):
    
    input_batches = []
    target_batches = []
    # use when dataset are a single bunch
    if len(data_dirs)==1:
        all_input_ori = np.load(data_dirs[0] + 'input.npy')
        all_target_ori = pd.read_csv(data_dirs[0]+'stress.csv',index_col=0)
    else:
        for i in range(0,len(data_dirs)):
            if data_dirs[i]:
                input_batch = np.load(data_dirs[i] + 'input.npy')
                input_batches.append(input_batch)

                target_batch = pd.read_csv(data_dirs[i]+'stress.csv',index_col=0)
                target_batches.append(target_batch)
                
        all_input_ori = np.concatenate(input_batches,axis=0)
        all_target_ori = pd.concat(target_batches)
    
    all_input = all_input_ori[:,:,:,:,ch_start:ch_end]
    all_target = all_target_ori.iloc[:,1:7]/1000000 # convert units to MPa

    trainval_idx = np.load(result_dir+'train_index_'+save_name+'.npy')
    
    #trainval_input = all_input[trainval_idx]
    trainval_target = all_target.iloc[trainval_idx]

    test_input = np.delete(all_input,trainval_idx,0)
    test_target =  all_target.drop(index=trainval_target.index)
    
    return test_input,test_target

def test_dataset_gen_15_45(save_name,data_dirs,result_dir,ch_start,ch_end):
    
    input_batches = []
    target_batches = []
    # use when dataset are a single bunch
    if len(data_dirs)==1:
        all_input_ori = np.load(data_dirs[0] + 'input.npy')
        all_target_ori = pd.read_csv(data_dirs[0]+'stress.csv',index_col=0)
    else:
        for i in range(0,len(data_dirs)):
            if data_dirs[i]:
                input_batch = np.load(data_dirs[i] + 'input.npy')
                input_batches.append(input_batch)

                target_batch = pd.read_csv(data_dirs[i]+'stress.csv',index_col=0)
                target_batches.append(target_batch)
                
        all_input_ori = np.concatenate(input_batches,axis=0)
        all_target_ori = pd.concat(target_batches)
    
    all_input = all_input_ori[:,:,:,:,ch_start:ch_end]
    all_target = all_target_ori.iloc[:,0:6] # convert units to MPa

    trainval_idx = np.load(result_dir+'train_index_'+save_name+'.npy')
    
    #trainval_input = all_input[trainval_idx]
    trainval_target = all_target.iloc[trainval_idx]

    test_input = np.delete(all_input,trainval_idx,0)
    test_target =  all_target.drop(index=trainval_target.index)
    
    return test_input,test_target

def new_test_dataset_gen(data_dirs,ch_start,ch_end):
       
    input_batches = []
    target_batches = []
    
    for i in range(0,len(data_dirs)):
        if data_dirs[i]:
            input_batch = np.load(data_dirs[i] + 'input.npy')
            input_batches.append(input_batch)
            
            target_batch = pd.read_csv(data_dirs[i]+'stress.csv',index_col=0)
            target_batches.append(target_batch)
            
    test_input_ori = np.concatenate(input_batches,axis=0)
    test_target_ori = pd.concat(target_batches)
    
    test_input = test_input_ori[:,:,:,:,ch_start:ch_end]
    test_target = test_target_ori.iloc[:,1:7]/1000000 # convert units to MPa
    test_idx = test_target_ori.iloc[:,0]
    
    return test_idx,test_input,test_target

def NormalizeData(data):
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    return data,data_min,data_max

def DeNormalizeData(data,data_min,data_max):
    data = data*(data_max-data_min)+data_min
    return data

def train_dataset_gen_cGAN(save_name,data_dirs,train_ratio,ch_start,ch_end,norm=True):

    input_batches = []
    target_batches = []
    
    for i in range(0,len(data_dirs)):
        if data_dirs[i]:
            input_batch = np.load(data_dirs[i] + 'input.npy')
            input_batches.append(input_batch)
            
            target_batch = np.load(data_dirs[i] + 'target.npy')
            target_batches.append(target_batch)
            
    all_input = np.concatenate(input_batches,axis=0)
    all_target = np.concatenate(target_batches,axis=0)
    
    # convert dtype from float64 to float32 to match the generator
    all_input = np.float32(all_input[:,:,:,:,ch_start:ch_end])
    
    # normalize target if normalization is used. Otherwise assign 0 to data_min
    # and data_max as they will not be used.
    if norm:
        all_target,data_min,data_max = NormalizeData(all_target)
    else:
        data_min=0
        data_max=0
    all_target = np.float32(all_target)

    
    # ----------------------- loading data finishes -------------------------------------#
    
    trainval_size = int(len(all_input)*train_ratio)
    train_size = int(trainval_size*1)
    val_size = trainval_size-train_size
    #test_size = len(all_input)-trainval_size
    
    # ------------ sepearte data randomly to train, val and test ----------------#
    
    trainval_idx = random.sample(range(0,len(all_target)),trainval_size)
    trainval_input = all_input[trainval_idx]
    trainval_target = all_target[trainval_idx]
    
    test_input = np.delete(all_input,trainval_idx,0)
    test_target =  np.delete(all_target,trainval_idx,0)
    
    val_idx = random.sample(range(0,len(trainval_target)),val_size)
#    val_input = trainval_input[val_idx]
#    val_target = trainval_target[val_idx]
    
    train_input = np.delete(trainval_input,val_idx,0)
    train_target = np.delete(trainval_target,val_idx,0)

    # save the index for testing
    np.save('train_index_'+save_name,trainval_idx)
    
    return train_input,train_target,test_input,test_target,data_min,data_max

def test_dataset_gen_cGAN(save_name,data_dirs,result_dir):
    
    input_batches = []
    target_batches = []
    
    for i in range(0,len(data_dirs)):
        if data_dirs[i]:
            input_batch = np.load(data_dirs[i] + 'input.npy')
            input_batches.append(input_batch)
            
            target_batch = np.load(data_dirs[i] + 'target.npy')
            target_batches.append(target_batch)
            
    all_input_ori = np.concatenate(input_batches,axis=0)
    all_target_ori = np.concatenate(target_batches,axis=0)
    
    # convert dtype from float64 to float32 to match the generator
    all_input = np.float32(all_input_ori[:,:,:,:,1:5])
    all_target,data_min,data_max = NormalizeData(all_target_ori)
    all_target = np.float32(all_target)

    trainval_idx = np.load(result_dir+'train_index_'+save_name+'.npy')

    test_input = np.delete(all_input,trainval_idx,0)
    test_target = np.delete(all_target,trainval_idx,0)
    
    return test_input,test_target

## Added because .removesuffix only works for python 3.9+
def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def remove_prefix(input_string, prefix):
    if prefix and input_string.startswith(prefix):
        return input_string[len(prefix):]
    return input_string

# Move data from simulation results to dataset folder
def move_data():
    data_types = ['strain','strain-eq','strain-pl','strain-pl-eq']
    typeid = 1
    for grain in [10,15,20,25,30,35,40,45,50]:
        dir_strain_cal = '/home/xiao/projects/inverse_mat_des/Simulation/strain_recal/'
        dir_target = '/home/xiao/projects/inverse_mat_des/ML/dataset/{0}_cGAN/{1}/'.format(grain,data_types[typeid])
        files_strain_1 = glob.glob(dir_strain_cal + "{0}/**/*{1}.step2".format(grain,data_types[typeid]), recursive = True)
        files_strain_2 = glob.glob(dir_strain_cal + "{0}_2/**/*{1}.step2".format(grain,data_types[typeid]), recursive = True)
        for file_strain_1 in files_strain_1:
            idx = remove_suffix(file_strain_1[(len(dir_strain_cal)+3):],'_sim.sim/results/elts/{0}/{0}.step2'.format(data_types[typeid])).replace('/','_')
            shutil.copy(file_strain_1,dir_target+idx+'_{0}.step2'.format(data_types[typeid]))
        for file_strain_2 in files_strain_2:
            idx = remove_suffix(file_strain_2[(len(dir_strain_cal)+5):],'_sim.sim/results/elts/{0}/{0}.step2'.format(data_types[typeid])).replace('/','_')
            shutil.copy(file_strain_2,dir_target+idx+'_{0}.step2'.format(data_types[typeid]))