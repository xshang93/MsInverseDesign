#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:11:50 2023

@author: xiao

NSGA-iii code for optimization

"""

import signal
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
import matplotlib.pyplot as plt
from data_preprocessing import DeNormalizeData
from scipy.stats import norm
from image_vis_3D import plot_3D_res

from joblib import load
from image_gen_3D import image_gen_3D_BO
from image_vis_3D import plot_3D
from yield_cal import yield_strength_cal
from cGAN_3D import Generator, Discriminator

from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.core.callback import Callback
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.indicators.hv import Hypervolume
from pymoo.operators.crossover.pntx import SinglePointCrossover



#initialize gen to 0
gen=0
# define the customized function, i.e., the trained CNN and cGAN models to be optimized
def stress_estimator(solution,type_obj):
        
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
    global sh_dir
    sh_dir = './opt_run/NSGA/'+run_name+'/GA_Gen_{0}_Sol_{1}_{2}/'.format(gen,counter,type_obj)

    if (os.path.exists(sh_dir))!=1:
        os.makedirs(sh_dir)
    
    # =========== Call MATLAB function to generate grain orientations ==============================#
    # input numbers can't be integer. Should be float. e.g. 10.0 instead of 10
    # nargout=0 signifies no output is being returned
    eng.data_gen(n_priorBeta,oriBeta_inp,n_colonies_max_glb,lamwidth_beta,lam_ratio,ms_id,run_name,ori_id,sh_dir,nargout=0) 
    
    # ========== Call Neper to generate microstructures ============================================#
    # counter = counter+1
    with open(sh_dir+'seeds','w') as seeds_file:
        for seed in seed_list:
            seeds_file.write(' '.join(map(str,seed))+'\n')
    sh_name = sh_dir+'generate_tess.sh'
    # make generate_tess file excutable
    st = os.stat(sh_name)    

    os.chmod(sh_name, st.st_mode | stat.S_IEXEC)
#    subprocess.run([sh_name], stderr = subprocess.DEVNULL, cwd=sh_dir)
    
# ================================= Error handling ================================================#
    
    try:
        # no bash output
        proc = subprocess.Popen([sh_name], stderr = subprocess.DEVNULL,stdout = subprocess.DEVNULL, cwd='./', shell=True, preexec_fn = os.setsid)
        # # with bash output
        # proc = subprocess.Popen([sh_name], cwd='./', shell=True, preexec_fn = os.setsid)
        proc.communicate(timeout=600)
    except TimeoutExpired:
        # proc.kill()
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        print('Tessellation timeout for {0}'.format(sh_dir))
        return 0,0          
    
#    print('tessellating {0}\n'.format(ms_id))
    try:
        images = image_gen_3D_BO(ms_id,sh_dir)
    except IndexError:
        print('Tessellation failed for {0}'.format(sh_dir))
        return 0,0
    # if alpha phase ratio exceed limits
    stgroup = glob.glob(sh_dir+'*.stgroup')
    with open(stgroup[0]) as f_stgroup:
        lines = f_stgroup.readlines()
        
    # ============ if the phase ratio is with range for Ti64 ========================================#        
    if float(lines[0])<0.28:
        print('Alpha phase ratio out of lower limit!')
        return 0,0
    elif float(lines[0])>0.80:
        print('Alpha phase ratio out of upper limit!')
        return 0,0

    # ============ Calculate global stress  ========================================================#        
    images = image_gen_3D_BO(ms_id,sh_dir)
    tf.get_logger().setLevel('ERROR')
    stress = model.predict(images) # predict global stress response
    
    # comment out if no std
    sc = load(result_dir+'std_scaler'+save_name+'.bin')
    stress = sc.inverse_transform(stress)
    # calculate the modulus and yield strength of this microstructure
    E,yield_strength = yield_strength_cal(stress)

    return stress,images

def E_cal(solution): 
    global stress_sol
    global images_sol
    stress_sol,images_sol = stress_estimator(solution,'E')
    if type(stress_sol) != int:
        E,yield_strength = yield_strength_cal(stress_sol)
        f = open(sh_dir+"E_sigma.txt", "a")
        f.write("E = {0}\nSigmay = {1}\n".format(E,yield_strength))
        f.close()
    else:
        return 10 # return a value that is for sure greater than current E
    # if obj_select==1:
    #     E=-E[0]
        
    return E[0]/100000
def sigmay_cal(solution): 
    # stress,images = stress_estimator(solution,'Sy')
    if type(stress_sol) != int:
        E,yield_strength = yield_strength_cal(stress_sol)
    else:
        return 10
    yield_strength[0]=1/(yield_strength[0]/1000)
    return yield_strength[0]
def kt_cal(solution):
    # stress,images = stress_estimator(solution,'Kt')
    global counter
    counter = counter+1
    global gen
    global pop_size
    gen = int(counter/pop_size)
    if type(stress_sol) == int:
        return 10
    else:
        #============ Calculate stress concentration factor kt =========================================#
        global generator
        pred = generator(images_sol[0].reshape(-1,32,32,32,4), training=True)
        pred = pred.numpy()
        pred = DeNormalizeData(pred,data_min,data_max)
        idx_max = np.unravel_index(np.argmax(pred), pred.shape)
        stress_max = pred[idx_max]
        stress_maxslice = pred[:,idx_max[1],:,:,:]
        stress_nom = np.average(stress_maxslice.flatten())
        kt = stress_max/stress_nom
        f = open(sh_dir+"Kt.txt", "a")
        f.write("Kt = {0}".format(kt))
        f.close()
    return kt    
    # #============ Output optimization results into a text file =============================#
    # with open(sh_dir+'result_values.txt','w') as f:
    #     f.write('E is ' + str(E)+'\n')
    #     f.write('Yield strength is '+str(yield_strength)+'\n')
    #     f.write('Optimization objective is '+str(opt_obj)+'\n')
    #     f.write('stress vector is '+str(stress)+'\n')
    #     f.write('best kt is {0}\n'.format(kt))
    #     plot_3D_res(pred/np.average(pred.reshape(-1,1)),stress_min=0,stress_max=1.400,image_format='svg',save_dir = sh_dir)
    #     np.save(sh_dir+'pred_best',pred)
    #     return E,yield_strength,opt_obj,stress,pred,kt
   
#============ global values relevant to GA function =============================#    
# ============================= selectors =====================================#
# object selector. 1--max sigmay and E; 2 --max sigmay but min E. Kt to be minimized in both cases
obj_select = 2
# if alpha phase ratio is fixed in GA
alpha = 1
# randomseed to initialize GA
random_seed = 4
# run_name to be saved
global run_name
run_name = 'NSGAiii_obj{0}_alpha{1}_rand{2}_kt'.format(obj_select,alpha,random_seed)

# ============================= constant values =====================================#

if (obj_select == 2):
    null_val = 1000 # set this value to one greater than objective function values. for obj 2 and 5, this is -10; for rest this is 0.
else:
    null_val=0
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
model = load(result_dir+'HP_result'+save_name+'.pkl').best_estimator_

# ----------------------Load the trained 3D-cGAN model ------------------------------------#
data_types = ['seq','strain','strain-eq','strain-pl','strain-pl-eq']
typeid=0

cgan_dir = './'
save_name_cgan = 'final_all_data_20221026'.format(data_types[typeid])
BATCH_SIZE = 1
min_max = np.load(cgan_dir+'min_max_'+save_name_cgan+'.npy')
data_min = min_max[0]
data_max = min_max[1]

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

# ----------------------start matlab engine ------------------------------------#
eng = matlab.engine.start_matlab()

# ---------------------- Set up the optimziation problem ------------------------------------#
objs = [
        lambda x: E_cal(x),
        lambda x: sigmay_cal(x),
        lambda x: kt_cal(x)
        ]

num_genes = int(n_priorBeta_glb)*6+alpha
n_var = num_genes
# assign different gene types
gene_type = []
xl_list = []
xu_list = []
for gene in range(num_genes):
    if gene<n_priorBeta_glb*3:
        gene_type.append(float)
        if gene<n_priorBeta_glb*2 and gene>=n_priorBeta_glb:
            xl_list.append(0)
            xu_list.append(180)
        else:
            xl_list.append(0)
            xu_list.append(360)
    elif alpha==1:
        if gene!=num_genes-1:
            gene_type.append(int)      
            xl_list.append(1)
            xu_list.append(29)
        else:
            gene_type.append(float)
            xl_list.append(0.1)
            xu_list.append(4)
    else:
        gene_type.append(int)      
        xl_list.append(1)
        xu_list.append(29)
# convert xl_list and xu_list to np arrays
xl = np.array(xl_list)
xu = np.array(xu_list)

problem = FunctionalProblem(n_var,
                            objs,
                            xl=xl,
                            xu=xu
                            )
# # ---------------------- For testing if the problem is well formulated ------------------ #
# test_data_dir = '/home/xiao/projects/inverse_mat_des/BO/20230803_rv1_25_obj4_alpha1_rand1_kt/Best_sol/'
# # first get all the angles
# a1 = []
# a2 = []
# a3 = []
# s1 = []
# s2 = []
# s3 = []
# for i in range(25):
#     f_angles=open(test_data_dir+'1_cell{0}'.format(i+1))
#     angles = f_angles.read()
#     angles_list = angles.split(' ')
#     a1.append(float(angles_list[0]))
#     a2.append(float(angles_list[1]))
#     a3.append(float(angles_list[2]))
# # then get all the seeds
# f_seeds=open(test_data_dir+'seeds')
# seeds = f_seeds.read()
# seeds = seeds.split('\n')
# for idx,seed in enumerate(seeds):
#     if idx !=25:
#         seed_list = seed.split(' ')
#         s1.append(float(seed_list[0])/0.03125)
#         s2.append(float(seed_list[1])/0.03125)
#         s3.append(float(seed_list[2])/0.03125)
# # lastly specify the alpha ratio
# a_ratio = [2.45921]

# # put them together
# test_inp = np.array(a1+a2+a3+s1+s2+s3+a_ratio)
# F = problem.evaluate(test_inp)
# # -------------------------------------------------------------------------------------- #


# ----------------------------- For running the optimization ------------------------- #
#size of each population
pop_size = 36
NSGA = 3
if NSGA==3:
# NSGA iii
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=6)
    crossover = SinglePointCrossover()
    algorithm = NSGA3(pop_size=pop_size,
                      ref_dirs=ref_dirs,
                      crossover=crossover,
                      )
else:
    algorithm = NSGA2(pop_size=pop_size)

class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.opt_0 = []


    def notify(self, algorithm):
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt_0.append(algorithm.opt[0].F)

class MyOutput(Output):

    def __init__(self):
        super().__init__()
        self.sol0 = Column("sol0", width=40)
        self.sol1 = Column("sol1", width=40)
        self.sol2 = Column("sol2", width=40)
        self.sol3 = Column("sol3", width=40)
        self.sol4 = Column("sol4", width=40)
        self.sol5 = Column("sol5", width=40)
        self.sol6 = Column("sol6", width=40)
        self.sol7 = Column("sol7", width=40)
        self.columns += [self.sol0,self.sol1,self.sol2,self.sol3,self.sol4,self.sol5,self.sol6,self.sol7]

    def update(self, algorithm):
        super().update(algorithm)
        self.sol0.set(algorithm.pop.get("F")[0])
        self.sol1.set(algorithm.pop.get("F")[1])
        self.sol2.set(algorithm.pop.get("F")[2])
        self.sol3.set(algorithm.pop.get("F")[3])
        self.sol4.set(algorithm.pop.get("F")[4])
        self.sol5.set(algorithm.pop.get("F")[5])
        self.sol6.set(algorithm.pop.get("F")[6])
        self.sol7.set(algorithm.pop.get("F")[7])



termination = DefaultMultiObjectiveTermination(
    n_max_gen=60,
    period=15
)

res = minimize(problem,
                algorithm,
                termination,
                seed=random_seed,
                callback=MyCallback(),
                output=MyOutput(),
                verbose=True,
                save_history=True)

X, F = res.opt.get("X", "F")
hist = res.history

n_evals = []             # corresponding number of function evaluations\
hist_F = []              # the objective space values in each generation
hist_cv = []             # constraint violation in each generation
hist_cv_avg = []         # average constraint violation in the whole population

for algo in hist:

    # store the number of function evaluations
    n_evals.append(algo.evaluator.n_eval)

    # retrieve the optimum from the algorithm
    opt = algo.opt

    # store the least contraint violation and the average in each population
    hist_cv.append(opt.get("CV").min())
    hist_cv_avg.append(algo.pop.get("CV").mean())

    # filter out only the feasible and append and objective space values
    feas = np.where(opt.get("feasible"))[0]
    hist_F.append(opt.get("F")[feas])

approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)

metric = Hypervolume(ref_point= np.array([1.25, 1.25, 1.6]),
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=approx_ideal,
                     nadir=approx_nadir)

save_dir = './opt_run/NSGA/{0}/'.format(run_name)
hv = [metric.do(_F) for _F in hist_F]

plt.figure(figsize=(7, 5))
plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
plt.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.savefig(save_dir+'hv_plot.svg')
plt.show()



np.save(save_dir+'results_x',res.X)
np.save(save_dir+'results_y',res.F)
np.save(save_dir+'n_evals',n_evals)
np.save(save_dir+'hv',hv)
np.save(save_dir+'hist_F',hist_F)
