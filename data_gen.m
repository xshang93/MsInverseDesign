% This script is used to loop and generate multiple numbers of 
% microstrutures with different grain sizes and lam_width for Ti64 for FEA 
% and ML training
% This is the version used for python library generation.

% Need to have MTEX installed

% Xiao Shang @ UofT, 20220721
% list of inputs:
% - n_priorBeta: number of prior Beta grains
% - n_colonies_max: maximum number of alpha grains per prior Beta grain
% - lamwidth_beta: with of beta lamella
% - lam_ratio: ratio of beta lamella
% - ms_ID: random seed for tessellation. i.e., -id in neper -T
% ----------------------------------------------------------------------- %

% addpath '/Users/Xiao/mtex-5.8.1'; % Point to whatever path that has MTEX
% 
% startup_mtex;

function data_gen(n_priorBeta,oriBeta_inp,n_colonies_max,lamwidth_beta,lam_ratio,ms_ID,run_name,ori_id,file_dir)
    % Where the files are generated locally and on hpc
%     data_dir_local = './'+string(run_name)+'/BO_ms_'+string(round(n_priorBeta))+'_'+string(round(n_colonies_max))+'_'+string(round(lamwidth_beta*100000))+'_'+string(round(lam_ratio*100000))+'_'+string(ms_ID)+'_'+string(ori_id)+'/';
%     if ~exist(data_dir_local, 'dir')
%        mkdir(data_dir_local)
%     end 
    % rng control the randomness of the orientation generation
    rng('default');
    %rng(ori_id); % ori_id is to control the generation of grain orientations
    rng(ms_ID); % set ori_id to ms_id to simplified the problem
    texture_strength = 0;
    n_colonies_min = n_colonies_max; %min number of colonies in each prior beta grain
    
    %file_dir = data_dir_local;

    ms_gen_25D_HPC(n_priorBeta,oriBeta_inp,n_colonies_min,n_colonies_max,lamwidth_beta,file_dir,ms_ID,lam_ratio,texture_strength);
end