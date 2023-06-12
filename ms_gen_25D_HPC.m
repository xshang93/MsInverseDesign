% This script is used for generate multiple numbers of microstrutures for
% Ti64 for FEA and ML training

% Need to have MTEX installed

% Xiao Shang @ UofT, 20211124
% This is the version used for python library generation.
% ----------------------------------------------------------------------- %

%addpath '/Users/Xiao/mtex-5.7.0'; % Point to whatever path that has MTEX

%startup_mtex;

function ms_gen_25D_HPC(n_priorBeta,oriBeta_inp,n_colonies_min,n_colonies_max,lamwidth_beta,file_dir,ms_ID,lam_ratio,texture_strength)
    
    %domain l/w/t
    l = 1;
    w = 1;
    t = 1;
    
    diameq_nu = 1; %grain equivalent diameter, mean
    diameq_sigma = 0.2; %grain equivalent diameter, sigma
    sph_nu = 0.855; %sphericity, mean
    sph_sigma = 0.03; %sphericity, sigma
    % morpho_t1 = 'diameq:lognormal({0},{1})'.format(diameq_nu,diameq_sigma) % Specify morphological properties of the cells page 14 of neper doc lognormal(mean,sigma)
    % morpho_t2 = '1-sphericity:normal({0},{1})'.format(sph_nu, sph_sigma) % Specify morphological properties of the cells page 14 of neper doc
    morpho_t1 = sprintf('diameq:lognormal(%0.4f,%0.4f),sphericity:normal(%0.4f,%0.4f)',diameq_nu,diameq_sigma,sph_nu,sph_sigma);
    morpho_t2 = sprintf('diameq:lognormal(%0.4f,%0.4f),sphericity:normal(%0.4f,%0.4f)',diameq_nu,diameq_sigma,sph_nu,sph_sigma);
    
    lamwidth_alpha = lamwidth_beta * lam_ratio; % alpha lamellar width, alpha:beta = 1:lam_ratio
    lamwidth_single = l*10; % width when alpha colony is a single grain, i.e., w/o lameller
    r_lameller = 0.2/0.2; % ratio of alpha colonies with lamellar

    % Gnerate .sh script for Neper tessellation
    script_tess = fopen(string(file_dir)+'generate_tess.sh', 'w');
    
    fprintf(script_tess,'#!/bin/bash\n');
    
    fprintf(script_tess,'# Shell script for geometry tessellation\n');
    fprintf(script_tess,'# - This bash script requires a full installation of Neper. Tested on v4.4.2-33\n');
    fprintf(script_tess,'cd '+string(file_dir)+'\n');
    
    ori_gen.ori_files(ms_ID,n_priorBeta,oriBeta_inp,n_colonies_min,n_colonies_max,file_dir,lamwidth_beta,lamwidth_alpha,lamwidth_single,r_lameller,texture_strength);% Generate orientation files

    % 3 scale tessellation (beta lamella)
    fprintf(script_tess,'# Generate beta lamella\n');
    n_grains = string(n_priorBeta)+'::'+'file('+string(ms_ID)+'_colonies)'+'::'+'from_morpho';
    morpho_t3 = string('lamellar(w=file('+string(ms_ID)+'_lamwidth),v=crysdir(-1,1,0),pos=optimal)');
    morpho = string(morpho_t1)+'::'+string(morpho_t2)+'::'+string(morpho_t3);
    if oriBeta_inp==0 % When ms id is used as input
        fprintf(script_tess,"neper -T -n '%s' -id %s -dim 3 -domain 'cube(%s,%s,%s)' -ori 'random::file(%s_scale2_ori)::random' -morpho '%s' -group 'lam==1?1:2' -statcell ""scaleid(1),lam"" -statgroup vol -reg 1 -sel %s -format tess -o %s_%s_%s_%s_%s\n",...
            [n_grains,string(ms_ID),string(l),string(w),string(t),string(ms_ID),morpho,string(lamwidth_beta*0.1), ...
            string(round(n_priorBeta)),string(round(n_colonies_max)),string(round(lamwidth_beta*100000)), ...
            string(round(lam_ratio*100000)),string(ms_ID)]);
    else
        fprintf(script_tess,"neper -T -n '%s' -id %s -morphooptiini 'coo:file(seeds)' -dim 3 -domain 'cube(%s,%s,%s)' -ori 'random::file(%s_scale2_ori)::random' -morpho '%s' -group 'lam==1?1:2' -statcell ""scaleid(1),lam"" -statgroup vol -reg 1 -sel %s -format tess -o %s_%s_%s_%s_%s\n",...
            [n_grains,string(ms_ID),string(l),string(w),string(t),string(ms_ID),morpho,string(lamwidth_beta*0.1), ...
            string(round(n_priorBeta)),string(round(n_colonies_max)),string(round(lamwidth_beta*100000)), ...
            string(round(lam_ratio*100000)),string(ms_ID)]);
    end

    fprintf(script_tess,"rm -f %s_scale3_ori\n",[string(ms_ID)]);
    for grain_ID_priorBeta = 1:1:n_priorBeta
        for grain_ID_colony = 1:1:n_colonies_max
            fprintf(script_tess,"awk '{if ($1==%s) print $2}' %s_%s_%s_%s_%s.stcell > %s_ori-grain%s\n", ...
                [string(grain_ID_priorBeta),string(round(n_priorBeta)),string(round(n_colonies_max)), ...
                string(round(lamwidth_beta*100000)),string(round(lam_ratio*100000)),string(ms_ID), ...
                string(ms_ID),string(grain_ID_priorBeta)]);
            fprintf(script_tess,"ORI_A=$(awk 'NR==1 {print $1,$2,$3}' %s_cell%s_%s)\n", ...
                [string(ms_ID),string(grain_ID_priorBeta),string(grain_ID_colony)]);
            fprintf(script_tess,"ORI_B=$(awk 'NR==2 {print $1,$2,$3}' %s_cell%s_%s)\n", ...
                [string(ms_ID),string(grain_ID_priorBeta),string(grain_ID_colony)]);
            fprintf(script_tess,"sed -i ""s/^\\<1\\>$/$ORI_A/"" %s_ori-grain%s\n", ...
                [string(ms_ID),string(grain_ID_priorBeta)]);
            fprintf(script_tess,"sed -i ""s/^\\<2\\>$/$ORI_B/"" %s_ori-grain%s\n", ...
                [string(ms_ID),string(grain_ID_priorBeta)]);
            fprintf(script_tess,"echo ""%s::%s file(%s_ori-grain%s,des=euler-bunge)"" >> %s_scale3_ori\n", ...
                [string(grain_ID_priorBeta),string(grain_ID_colony),string(ms_ID),string(grain_ID_priorBeta),string(ms_ID)]);
        end
    end
    if oriBeta_inp==0 % When ms id is used as input
        fprintf(script_tess,"neper -T -n '%s' -id %s -dim 3 -domain 'cube(%s,%s,%s)' -ori 'random::file(%s_scale2_ori)::file(%s_scale3_ori)' -morpho '%s' -group 'lam==1?1:2' -statgroup vol -reg 1 -sel %s -format tess -o %s_%s_%s_%s_%s\n",...
            [n_grains,string(ms_ID),string(l),string(w),string(t),string(ms_ID),string(ms_ID),morpho,string(lamwidth_beta*0.1), ...
            string(round(n_priorBeta)),string(round(n_colonies_max)),string(round(lamwidth_beta*100000)), ...
            string(round(lam_ratio*100000)),string(ms_ID)]);
        fprintf(script_tess,"neper -T -n '%s' -id %s -dim 3 -domain 'cube(%s,%s,%s)' -ori 'random::file(%s_scale2_ori)::file(%s_scale3_ori)' -morpho '%s' -reg 1 -sel %s -format tesr -tesrformat 'ascii' -tesrsize 32 -o %s_%s_%s_%s_%s\n",...
            [n_grains,string(ms_ID),string(l),string(w),string(t),string(ms_ID),string(ms_ID),morpho,string(lamwidth_beta*0.1), ...
            string(round(n_priorBeta)),string(round(n_colonies_max)),string(round(lamwidth_beta*100000)), ...
            string(round(lam_ratio*100000)),string(ms_ID)]);
    else
        fprintf(script_tess,"neper -T -n '%s' -id %s -morphooptiini 'coo:file(seeds)' -dim 3 -domain 'cube(%s,%s,%s)' -ori 'random::file(%s_scale2_ori)::file(%s_scale3_ori)' -morpho '%s' -group 'lam==1?1:2' -statgroup vol -reg 1 -sel %s -format tess -o %s_%s_%s_%s_%s\n",...
            [n_grains,string(ms_ID),string(l),string(w),string(t),string(ms_ID),string(ms_ID),morpho,string(lamwidth_beta*0.1), ...
            string(round(n_priorBeta)),string(round(n_colonies_max)),string(round(lamwidth_beta*100000)), ...
            string(round(lam_ratio*100000)),string(ms_ID)]);
        fprintf(script_tess,"neper -T -n '%s' -id %s -morphooptiini 'coo:file(seeds)' -dim 3 -domain 'cube(%s,%s,%s)' -ori 'random::file(%s_scale2_ori)::file(%s_scale3_ori)' -morpho '%s' -reg 1 -sel %s -format tesr -tesrformat 'ascii' -tesrsize 32 -o %s_%s_%s_%s_%s\n",...
            [n_grains,string(ms_ID),string(l),string(w),string(t),string(ms_ID),string(ms_ID),morpho,string(lamwidth_beta*0.1), ...
            string(round(n_priorBeta)),string(round(n_colonies_max)),string(round(lamwidth_beta*100000)), ...
            string(round(lam_ratio*100000)),string(ms_ID)]);
    end
    fprintf(script_tess,'#Generate geometry image \n');
    fprintf(script_tess,"neper -V %s_%s_%s_%s_%s.tess -datacellcol 'ori' -datacellcolscheme 'ipf(z)' -print %s_%s_%s_%s_%s_geom \n",...
        [string(round(n_priorBeta)),string(round(n_colonies_max)),string(round(lamwidth_beta*100000)), ...
        string(round(lam_ratio*100000)),string(ms_ID),string(round(n_priorBeta)),string(round(n_colonies_max)), ...
        string(round(lamwidth_beta*100000)),string(round(lam_ratio*100000)),string(ms_ID)]);
    fprintf(script_tess,"neper -V %s_%s_%s_%s_%s.tesr -datacellcol 'ori' -datacellcolscheme 'ipf(z)' -print %s_%s_%s_%s_%s_geom_raster \n\n",...
        [string(round(n_priorBeta)),string(round(n_colonies_max)),string(round(lamwidth_beta*100000)), ...
        string(round(lam_ratio*100000)),string(ms_ID),string(round(n_priorBeta)),string(round(n_colonies_max)), ...
        string(round(lamwidth_beta*100000)),string(round(lam_ratio*100000)),string(ms_ID)]);
    
    fprintf(script_tess,'exit 0');
    fclose(script_tess);
end

