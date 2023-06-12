#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:11:20 2022

@author: Xiao

Used to generate 3D images (arrays) with multiple input channels for BO

"""

import numpy as np
import glob
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

# generate input for BO
def image_gen_3D_BO(ms_id,sh_dir):
    
    # # ------------------------ Move raw results files end ------------------------------#
           
    raster_size = 32
    n_channels = 5
    
    images = []
      
    try:     
    # ------------------------- gather 3D image data ---------------------------#   
        f_tess = open(glob.glob(sh_dir.replace('generate_tess.sh','')+'*.tess')[0])    
        lines_tess = f_tess.readlines()
        f_tesr = open(glob.glob(sh_dir.replace('generate_tess.sh','')+'*.tesr')[0])    
        lines_tesr = f_tesr.readlines()
        
        cellids_list = []
        ori_list = []
        
        for line_number,line in enumerate(lines_tess):
            # First find how many cells are in the microstructure
            if '**cell' in line:
                num_cells = int(lines_tess[line_number+1])
            
            # Then find and store lam info
            if '*lam' in line:
                lam_line_start = line_number+1
            if '*group' in line:
                lam_line_end = line_number
                lam_list=[]        
                for i in range(lam_line_start,lam_line_end):
                    lams_str = lines_tess[i]
                    for lam_str in lams_str.split():
                        lam_int = int(lam_str)-1 #convert 1,2 to binary (0,1)
                        lam_list.append(lam_int)
            # Then extract the orientations and store them in a list
            if '*ori' in line:
                for ori in range(line_number+2,line_number+2+num_cells):
                    r1 = float(lines_tess[ori][2:17])
                    r2 = float(lines_tess[ori][20:35])
                    r3 = float(lines_tess[ori][38:53])
                    r = [r1,r2,r3]
                    ori_list.append(r)
                    
        # Finally extract the cell ids for each voxel from the .tesr file
        for line_number,line in enumerate(lines_tesr):
            if 'ascii\n' in line:
                for i in range(line_number+1,len(lines_tesr)-1):
                    cellids_str = lines_tesr[i]
                    for cellid_str in cellids_str.split():
                        cellid_int = int(cellid_str)
                        cellids_list.append(cellid_int)
        
        image = np.zeros((raster_size,raster_size,raster_size,n_channels))
        counter=0
        while counter < len(cellids_list):
            for z in range(0,raster_size):
                for y in range(0,raster_size):
                    for x in range(0,raster_size):
                        # Channel 0 is cell id
                        image[x,y,z,0] = cellids_list[counter]
                        # Channel 0 is lam id
                        image[x,y,z,1] = lam_list[cellids_list[counter]-1]
                        # Channels 2-3 are orientations
                        image[x,y,z,2:n_channels] = ori_list[cellids_list[counter]-1]
                        counter = counter+1
                        
    # ------------------------- gather 3D image data to files ---------------------------#   
        images.append(image[:,:,:,1:5])
    except ValueError:
        print('ValueError:failed!')
    
    return images

# generate input files for 3DCNN
def image_gen_3D_CNN(dir_original,dir_raw_input,dir_raw_target,dir_preprocessed,move,gen):
    
#    dir_original = '/media/xiao/One_Touch/Xiao/Simulation/hpc_20220621_25grains/texture2/'
#    dir_raw_input = '/home/xiao/projects/inverse_mat_des/ML/dataset/25_t2/raw/input/'
#    dir_raw_target = '/home/xiao/projects/inverse_mat_des/ML/dataset/25_t2/raw/target/'
#    dir_preprocessed = '/home/xiao/projects/inverse_mat_des/ML/dataset/25_t2/preprocessed/'
#   
    # if raw data needs to be moved, set move=1
    # ------------------------ Move raw results files ------------------------------#
    files_tesr = glob.glob(dir_original + "**/*.tesr", recursive = True)
    files_tess = glob.glob(dir_original + "**/*.tess", recursive = True)
    files_force = glob.glob(dir_original + "**/*.x1", recursive = True)
    if move:

        
        for file_tesr in files_tesr:
            file_new = shutil.copy(file_tesr,dir_raw_input)
            file_new_name = dir_raw_input+file_tesr[len(dir_original):].replace('/','_')
            os.rename(file_new,file_new_name)
        
        for file_tess in files_tess:
            file_new = shutil.copy(file_tess,dir_raw_input)
            file_new_name = dir_raw_input+file_tess[len(dir_original):].replace('/','_')
            os.rename(file_new,file_new_name)
            
        for file_force in files_force:
            file_new = shutil.copy(file_force,dir_raw_target)
            file_new_name = dir_raw_target+file_force[len(dir_original):].replace('/','_')
            os.rename(file_new,file_new_name)
    
    # if results is to be generated, set gen=1
    if gen:
        # ------------------------ Move raw results files end ------------------------------#
            
        files_force_new = glob.glob(dir_raw_target + "**/*.x1", recursive = True)
               
        raster_size = 32
        n_channels = 5
        col_names = ['step', 'INCR', 'Fx', 'Fy', 'Fz', 'area A', 'TIME']
        area = 0.000001
        strain = np.linspace(0,0.015,7)
        interp_factor = 10 # density of interpolation. larger the finer
        strain_new = np.linspace(0,0.015,7*interp_factor)
        data_E = []; data_yield = []
        
        images = []
        idx_list = []
        data_stress = []
        
        counter_file = 0
        for file_force_new in files_force_new:  
            try:
                # ------------------------- gather stress/E data ---------------------------#    
                idx = remove_suffix(file_force_new.replace(dir_raw_target,''),'_post.force.x1')
                # load the data and pick Fx
                data = pd.read_csv(file_force_new,skiprows = [0,1], delimiter='   ', names=col_names, engine='python')
                fx = data['Fx'].tolist()
                stress = [f/area for f in fx] # calculate engineering stress from reaction force
                E = (stress[1]/0.0025+stress[2]/0.005)/2 # calculate Young's modulus, average
                intercept = -E*0.002 # calculate the intercept of the 0.2% offsetline
                model_interp = interpolate.interp1d(strain,stress,'linear') # fit the current interpolation model
                stress_new = model_interp(strain_new) # interpolate the stress
                offsetline = E*strain_new+intercept # assemble the offset line
                
                idx_yield = np.argwhere(np.diff(np.sign(stress_new - offsetline))).flatten() # find the index of the yield stress
                yield_strength = stress_new[idx_yield][0] # '0' is used to convert array to float number
                
                # Plot data every 200 samples
                if counter_file%200 == 0:
                    plt.plot(strain_new,stress_new,'-b')
                    plt.plot(strain_new,offsetline,'-g')
                    plt.plot(strain_new[idx_yield], stress_new[idx_yield], 'ro')
                    
                counter_file+=1
                
            # ------------------------- gather 3D image data ---------------------------#   
                f_tess = open(glob.glob(dir_raw_input+idx+'*.tess')[0])    
                lines_tess = f_tess.readlines()
                f_tesr = open(glob.glob(dir_raw_input+idx+'*.tesr')[0])    
                lines_tesr = f_tesr.readlines()
                
                cellids_list = []
                ori_list = []
                
                for line_number,line in enumerate(lines_tess):
                    # First find how many cells are in the microstructure
                    if '**cell' in line:
                        num_cells = int(lines_tess[line_number+1])
                    
                    # Then find and store lam info
                    if '*lam' in line:
                        lam_line_start = line_number+1
                    if '*group' in line:
                        lam_line_end = line_number
                        lam_list=[]        
                        for i in range(lam_line_start,lam_line_end):
                            lams_str = lines_tess[i]
                            for lam_str in lams_str.split():
                                lam_int = int(lam_str)-1 #convert 1,2 to binary (0,1)
                                lam_list.append(lam_int)
                    # Then extract the orientations and store them in a list
                    if '*ori' in line:
                        for ori in range(line_number+2,line_number+2+num_cells):
                            r1 = float(lines_tess[ori][2:17])
                            r2 = float(lines_tess[ori][20:35])
                            r3 = float(lines_tess[ori][38:53])
                            r = [r1,r2,r3]
                            ori_list.append(r)
                            
                # Finally extract the cell ids for each voxel from the .tesr file
                for line_number,line in enumerate(lines_tesr):
                    if 'ascii\n' in line:
                        for i in range(line_number+1,len(lines_tesr)-1):
                            cellids_str = lines_tesr[i]
                            for cellid_str in cellids_str.split():
                                cellid_int = int(cellid_str)
                                cellids_list.append(cellid_int)
                
                image = np.zeros((raster_size,raster_size,raster_size,n_channels))
                counter=0
                while counter < len(cellids_list):
                    for z in range(0,raster_size):
                        for y in range(0,raster_size):
                            for x in range(0,raster_size):
                                # Channel 0 is cell id
                                image[x,y,z,0] = cellids_list[counter]
                                # Channel 1 is lam id
                                image[x,y,z,1] = lam_list[cellids_list[counter]-1]
                                # Channels 2-3 are orientations
                                image[x,y,z,2:5] = ori_list[cellids_list[counter]-1]
                                counter = counter+1
                                
            # ------------------------- gather 3D image data to files ---------------------------#   
            
                data_yield.append(yield_strength)
                data_E.append(E) # add the modulus for this sample into the dataset
                idx_list.append(idx) # put the index into the index vector
                data_stress.append(stress) # put the stress vector for this sample into the dataset
                
                images.append(image)
            except ValueError:
                print(file_force_new+'failed!')
            #except IndexError:
            #    print(file_force_new+'failed!')
                
        stress_df = pd.DataFrame(data_stress, columns = ['0.00','0.25','0.50','0.75','1.00','1.25','1.50']) # pd dataframe storing the stress values
        stress_df['E'] = data_E
        stress_df['Yield'] = data_yield
        
        stress_df.index = idx_list
        
        stress_df.to_csv(dir_preprocessed + 'stress.csv')
            
        np.save(dir_preprocessed + 'input',images)# save all 3D images into a .npy file
            
        return 

# Generate the position file for extracting mesh elements
def position_gen():
    x_coo = np.linspace(0.015,0.985,32)
    y_coo = np.linspace(0.015,0.985,32)
    z_coo = np.linspace(0.015,0.985,32)
    raster_size = 32
    # write position file, only use once
    f = open('./positions', 'w')
    for x in range(0,raster_size):
        for y in range(0,raster_size):
            for z in range(0,raster_size):
                f.write('{0} {1} {2}\n'.format(x_coo[x],y_coo[y],z_coo[z]))
                
    f.close()
## Added because .removesuffix only works for python 3.9+
def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

## generate output stress/strain 3D images for cGAN
def image_gen_3D_cGAN(typeid=4):
    # For selecting target data type to deal with
    data_types = ['seq','strain','strain-eq','strain-pl','strain-pl-eq']
    
    for grain in [10,15,20,25,30,35,40,45,50]:
        # target property data directory, selected by data_types
        dir_raw = '/home/xiao/projects/inverse_mat_des/ML/dataset/'+str(grain)+'_cGAN/{0}/'.format(data_types[typeid])
        # stpoint file directory
        dir_stpoint = '/home/xiao/projects/inverse_mat_des/ML/dataset/'+str(grain)+'_cGAN/stpoint/'
        # .tesr geometry files directory
        dir_raw_input = '/home/xiao/projects/inverse_mat_des/ML/dataset/all_input/'
        # output directory, create if not exists already
        dir_preprocessed = '/home/xiao/projects/inverse_mat_des/ML/dataset/'+str(grain)+'_cGAN/preprocessed_{0}/'.format(data_types[typeid])
        isExists = os.path.exists(dir_preprocessed)
        if isExists==0:
            os.makedirs(dir_preprocessed)

        images_tar = []
        images_inp = []
        files_tar_new = glob.glob(dir_raw + "*{0}.step2".format(data_types[typeid]), recursive = True)
        idx_list = []
        for file_tar_new in files_tar_new:
            try:
                idx = remove_suffix(file_tar_new.replace(dir_raw,''),'_{0}.step2'.format(data_types[typeid]))
                # stpoint file name sample '/home/xiao/projects/inverse_mat_des/ML/dataset/10_cGAN/preprocessed/20_2_20_1_19_12_0_ori1_27.stpoint'
                stpoint = np.loadtxt(dir_stpoint+idx+'.stpoint')
                tar = np.loadtxt(file_tar_new)
                raster_size = 32
                
                image_tar = np.zeros((32,32,32,1))
                
                counter=0
                for x in range(0,raster_size):
                    for y in range(0,raster_size):
                        for z in range(0,raster_size):    
                            image_tar[x][y][z] = tar[int(stpoint[counter])-1]
                            counter=counter+1                            
        
                f_tess = open(glob.glob(dir_raw_input+idx+'*.tess')[0])    
                lines_tess = f_tess.readlines()
                f_tesr = open(glob.glob(dir_raw_input+idx+'*.tesr')[0])    
                lines_tesr = f_tesr.readlines()
                
                cellids_list = []
                ori_list = []
                
                for line_number,line in enumerate(lines_tess):
                    # First find how many cells are in the microstructure
                    if '**cell' in line:
                        num_cells = int(lines_tess[line_number+1])
                    
                    # Then find and store lam info
                    if '*lam' in line:
                        lam_line_start = line_number+1
                    if '*group' in line:
                        lam_line_end = line_number
                        lam_list=[]        
                        for i in range(lam_line_start,lam_line_end):
                            lams_str = lines_tess[i]
                            for lam_str in lams_str.split():
                                lam_int = int(lam_str)-1 #convert 1,2 to binary (0,1)
                                lam_list.append(lam_int)
                    # Then extract the orientations and store them in a list
                    if '*ori' in line:
                        for ori in range(line_number+2,line_number+2+num_cells):
                            r1 = float(lines_tess[ori][2:17])
                            r2 = float(lines_tess[ori][20:35])
                            r3 = float(lines_tess[ori][38:53])
                            r = [r1,r2,r3]
                            ori_list.append(r)
                            
                # Finally extract the cell ids for each voxel from the .tesr file
                for line_number,line in enumerate(lines_tesr):
                    if 'ascii\n' in line:
                        for i in range(line_number+1,len(lines_tesr)-1):
                            cellids_str = lines_tesr[i]
                            for cellid_str in cellids_str.split():
                                cellid_int = int(cellid_str)
                                cellids_list.append(cellid_int)
                n_channels = 5
                image_inp = np.zeros((raster_size,raster_size,raster_size,n_channels))
                counter=0
                while counter < len(cellids_list):
                    for z in range(0,raster_size):
                        for y in range(0,raster_size):
                            for x in range(0,raster_size):
                                # Channel 0 is cell id
                                image_inp[x,y,z,0] = cellids_list[counter]
                                # Channel 1 is lam id
                                image_inp[x,y,z,1] = lam_list[cellids_list[counter]-1]
                                # Channels 2-3 are orientations
                                image_inp[x,y,z,2:5] = ori_list[cellids_list[counter]-1]
                                counter = counter+1
                               
                images_tar.append(image_tar)
                images_inp.append(image_inp)
                idx_list.append(idx)
            except IndexError:
                print(idx+' failed!')
            except FileNotFoundError:
                print(idx+' failed!')
            
        np.save(dir_preprocessed + 'target',images_tar)# save all 3D images into a .npy file
        np.save(dir_preprocessed + 'input',images_inp)# save all 3D images into a .npy file
        np.save(dir_preprocessed + 'idx',idx_list)# save all indexes into a .npy file
            
                


