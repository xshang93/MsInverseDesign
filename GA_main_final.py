#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:54:43 2022

@author: Xiao

Bayesian opt test

"""
import subprocess
from subprocess import TimeoutExpired
import matlab.engine
import os
import stat
import tensorflow as tf
#import sklearn
import numpy as np
import glob

import model_builder_3DCNN
#import matplotlib.pyplot as plt
from data_preprocessing import DeNormalizeData
#from scipy.stats import norm
from image_vis_3D import plot_3D_res

from joblib import load
from image_gen_3D import image_gen_3D_BO
#from image_vis_3D import plot_3D
from yield_cal import yield_strength_cal
from cGAN_3D import Generator, Discriminator

import pygad

# define the customized function, i.e., the trained CNN model to be optimized
def stress_estimator(solution,solution_idx):
    
    global counter
    global ga_instance
    only_obj = only_obj_glb
    
    gen = ga_instance.generations_completed
    
    n_priorBeta = float(int(n_priorBeta_glb))
    ms_id = float(int(ms_id_glb))
    ori_id = float(int(ori_id_glb))
    lamwidth_beta = float(lamwidth_beta_glb)
    if alpha ==1:
        lam_ratio = float(solution[-1])
    else:
        lam_ratio = float(lam_ratio_glb)
    
    # Create seed list from solution input
    ori_list = []
    seed_list = []
    for i in range(int(n_priorBeta)):
       ori_list.append([solution[i],solution[i+int(n_priorBeta)],solution[i+int(n_priorBeta*2)]]) 
       seed_list.append([solution[i+int(n_priorBeta*3)]*0.03125,solution[i+int(n_priorBeta*4)]*0.03125,solution[i+int(n_priorBeta*5)]*0.03125]) 
    
    oriBeta_inp = matlab.double(ori_list)
#    seed_list_new = seed_list
#    for i,seed in enumerate(seed_list):
#        for j,coord in enumerate(seed):
#            if coord%0.03125!=0:
#                seed_list_new[i][j] = round(coord)*0.03125
    
    # ========= Decide whether to run the optimization or just visulizing the best result ==========#
    if only_obj:
        sh_dir = './opt_run/GA_Gen_{0}_Sol_{1}/'.format(gen,counter)
    else:
        sh_dir = './opt_run/Best_sol/'
    if (os.path.exists(sh_dir))!=1:
        os.makedirs(sh_dir)
    
    # =========== Call MATLAB function to generate grain orientations ==============================#
    # input numbers can't be integer. Should be float. e.g. 10.0 instead of 10
    # nargout=0 signifies no output is being returned
    eng.data_gen(n_priorBeta,oriBeta_inp,n_colonies_max_glb,lamwidth_beta,lam_ratio,ms_id,run_name,ori_id,sh_dir,nargout=0) 
    
    # ========== Call Neper to generate microstructures ============================================#
    counter = counter+1
    with open(sh_dir+'seeds','w') as seeds_file:
        for seed in seed_list:
            seeds_file.write(' '.join(map(str,seed))+'\n')
    sh_name = sh_dir+'generate_tess.sh'
    # make generate_tess file excutable
    st = os.stat(sh_name)    

    os.chmod(sh_name, st.st_mode | stat.S_IEXEC)
#    subprocess.run([sh_name], stderr = subprocess.DEVNULL, cwd=sh_dir)
    
# ================================= Error handling ================================================#
    global lower_bd
    
    try:
#        subprocess.run([sh_name], stderr = subprocess.DEVNULL, cwd=sh_dir,timeout=600)
        proc = subprocess.Popen([sh_name], stdout=subprocess.DEVNULL, stderr = subprocess.STDOUT, cwd='./')
        proc.communicate(timeout=600)
    except TimeoutExpired:
        proc.kill()
        print('Tessellation timeout for {0}'.format(sh_dir))
        return lower_bd          
    
#    print('tessellating {0}\n'.format(ms_id))
    try:
        images = image_gen_3D_BO(ms_id,sh_dir)
    except IndexError:
        print('Tessellation failed for {0}'.format(sh_dir))
        return lower_bd
    # if alpha phase ratio exceed limits
    stgroup = glob.glob(sh_dir+'*.stgroup')
    with open(stgroup[0]) as f_stgroup:
        lines = f_stgroup.readlines()
        
    # ============ if the phase ratio is with range for Ti64 ========================================#        
    if float(lines[0])<0.15:
        print('Alpha phase ratio out of lower limit!')
        return lower_bd
    elif float(lines[0])>0.95:
        print('Alpha phase ratio out of upper limit!')
        return lower_bd
    
    # ============ Calculate global stress  ========================================================#        
    images = image_gen_3D_BO(ms_id,sh_dir)
    tf.get_logger().setLevel('ERROR')
    stress = model.predict(images) # predict global stress response
    
    # comment out if no std
    sc = load(result_dir+'std_scaler'+save_name+'.bin')
    stress = sc.inverse_transform(stress)
    # calculate the modulus and yield strength of this microstructure
    E,yield_strength = yield_strength_cal(stress)
    
    #============ Calculate stress concentration factor kt =========================================#
    global generator
    pred = generator(images[0].reshape(-1,32,32,32,4), training=True)
    pred = pred.numpy()
    pred = DeNormalizeData(pred,data_min,data_max)
    idx_max = np.unravel_index(np.argmax(pred), pred.shape)
    stress_max = pred[idx_max]
    stress_maxslice = pred[:,idx_max[1],:,:,:]
    stress_nom = np.average(stress_maxslice.flatten())
    kt = stress_max/stress_nom
    #mu, std = norm.fit(pred.reshape(-1,1)) # fit the predictions into a normal distribution
    
    #===================== calculate the optimization objects ======================================#
    global obj_select
    global tar_E
    global tar_sigmay
    
    # maximize E and yield_strength
    if obj_select==0: 
        opt_obj = E[0]/100000+yield_strength[0]/1000 
        
    # minimize E and maximize yield_strength
    elif obj_select==1:
        opt_obj = -E[0]/100000+yield_strength[0]/1000 
        
    # minimize E and fix yield_strength
    elif obj_select==2: 
        if tar_sigmay>yield_strength[0]: # i.e., 964<YS
            opt_obj = 1/(E[0]/100000)*0.8 # 0.8 is a penalty factor when tar_sigma is too far from current YS
        else:
            opt_obj = 1/(E[0]/100000) 
            
    # fix E and maximize yield_strength
    elif obj_select==3: 
        if abs(tar_E-E[0])>5000:
            print('E out of fixed range!')
            return lower_bd
        else:
            opt_obj = yield_strength[0]/1000-kt  
            
    # maximize E, yield, and minimize kt
    elif obj_select==4:
        opt_obj = E[0]/100000+yield_strength[0]/1000-kt
        
    # maximize yield strength, minimize E and kt
    elif obj_select==5:
        opt_obj = -E[0]/100000+yield_strength[0]/1000-kt
        
    print('Target value is {0}\n'.format(opt_obj))
    
    #============ Output optimization results into a text file =============================#
    # if only_obj==1, it's for GA optimization. If not, it's for calculating the best solution result only
    if only_obj:
        return opt_obj
    else:
        with open(sh_dir+'result_values.txt','w') as f:
            f.write('E is ' + str(E)+'\n')
            f.write('Yield strength is '+str(yield_strength)+'\n')
            f.write('Optimization objective is '+str(opt_obj)+'\n')
            f.write('stress vector is '+str(stress)+'\n')
            f.write('best kt is {0}\n'.format(kt))
            plot_3D_res(pred/np.average(pred.reshape(-1,1)),stress_min=0,stress_max=1.400,image_format='svg',save_dir = sh_dir)
            np.save(sh_dir+'pred_best',pred)
        return E,yield_strength,opt_obj,stress,pred,kt

def callback_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])

# ====================== Set optimization objectives ==================================== #
obj_select = 4 # 0 is max sigma y maxE; 1 max sigma y min E; 2 fix sigma; 3 fix E; 4 and 5 added kt


random_seed = 42
tar_sigmay = 964
tar_E = 105000
if (obj_select == 2) or (obj_select == 5):
    lower_bd = -2 # set this value to one smaller than objective function values. for obj 2 and 5, this is -10; for rest this is 0.
elif obj_select==3:
    lower_bd = -1
else:
    lower_bd=0

counter = 0 # to keep track of the iterations

ms_id_glb=1
ori_id_glb=1
n_priorBeta_glb=25.0
n_colonies_max_glb=1.0
lamwidth_beta_glb=0.15
lam_ratio_glb = 1.0


# ------------------------name and dir of saved 3DCNN ------------------------------------#
save_name = 'SGD_alldata_200eps'
result_dir = './'

# ----------------------Load the trained 3D-cGAN model ------------------------------------#
data_types = ['seq','strain','strain-eq','strain-pl','strain-pl-eq']
typeid=0

cgan_dir = './'
save_name_cgan = 'final_all_data_20221026'
BATCH_SIZE = 1
min_max = np.load(cgan_dir+'min_max_'+save_name_cgan+'.npy')
data_min = min_max[0]
data_max = min_max[1]

# --------------------------- Testing section ------------------------------ #
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

generator = Generator()
discriminator = Discriminator()

checkpoint_dir = cgan_dir+'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
 # restore any checkpoint
checkpoint_id = 3
checkpoint.restore(checkpoint_dir+'/ckpt-{0}'.format(checkpoint_id))

                    
# ----------------------name of the optimization run---------------------------------------#
alpha = 1 # 0 or 1. use phase ratio as optimization parameter or not
run_name = '20230415_{0}_obj{1}_alpha{2}_rand{3}_kt'.format(int(n_priorBeta_glb),obj_select,alpha,random_seed)
save_name_opt = run_name
model = load(result_dir+'HP_result'+save_name+'.pkl').best_estimator_

fitness_function = stress_estimator

num_generations = 60
num_parents_mating = 4

sol_per_pop = 8
num_genes = int(n_priorBeta_glb)*6+alpha

# assign different gene types
gene_type = []
gene_space = []
for gene in range(num_genes):
    if gene<n_priorBeta_glb*3:
        gene_type.append(float)
        if gene<n_priorBeta_glb*2 and gene>=n_priorBeta_glb:
            gene_space.append({'low':0,'high':180})
        else:
            gene_space.append({'low':0,'high':360})
    elif alpha==1:
        if gene!=num_genes-1:
                    gene_type.append(int)      
                    gene_space.append({'low':1,'high':29})
        else:
            gene_type.append(float)
            gene_space.append({'low':0.1,'high':4})
    else:
        gene_type.append(int)      
        gene_space.append({'low':1,'high':29})

#init_range_low = 0
#init_range_high = 4

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

# -------------------- Optimization loop -------------------------------------#
eng = matlab.engine.start_matlab()

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
#                       init_range_low=init_range_low,random_seed=2
#                       init_range_high=init_range_high,
                       gene_space = gene_space,
                       gene_type = gene_type,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=callback_gen,
                       save_best_solutions=True,
                       save_solutions=True,
                       random_seed=random_seed,
                       stop_criteria=["saturate_15"] # stop after the solution is not changing for 15 generations
                       )
# Run GA
only_obj_glb = 1 # this value needs to be 1 in the GA loops
ga_instance.run()
#
# Plot fitness
ga_instance.plot_fitness(
        title = 'Generation VS Fitness {0}'.format(run_name),
        xlabel = 'Fitness',ylabel = 'Generation',
        linewidth = 3,
        font_size = 14,
        plot_type = 'plot',
        color = "#3870FF",
        save_dir = './'+run_name+'/fitness_plot.svg'
        )


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

only_obj_glb = 0 # this value sets to be 0 when only calculating the results
E_best,yield_strength_best,opt_obj_best,stress_best,pred_best,kt_best = stress_estimator(solution,solution_idx)

eng.quit()

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

file_name = './'+run_name+'/result'
# save results
ga_instance.save(filename=file_name)

## load results
#ga_instance = pygad.load(filename=file_name)
#print(ga_instance.best_solution())