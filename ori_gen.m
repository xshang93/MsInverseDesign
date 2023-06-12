% This script is used for randomly generarte prior beta orientations, and
% resulting alpha colonies' orientations obeying the Burgers OR.
% Reference https://mtex-toolbox.github.io/ParentChildVariants.html
% This is the version used for python library generation.

% Need to have MTEX installed

% Xiao Shang @ UofT, 20220721
% This is the version used for python library generation.
% ----------------------------------------------------------------------- %
% Changed on 20221116. Take ori and seeds as input for tessellation.

classdef ori_gen
    methods (Static)
        function [] = ori_files(ms_ID, n_priorBeta,oriBeta_inp,n_colonies_min,n_colonies_max,path,lamwidth_beta,lamwidth_alpha,lamwidth_single,r_lameller,texture_strength)

            n_colonies_tofile = zeros(n_priorBeta,1); % to write to the colonie file, telling Neper the number of colonies in each prior beta grain
            oriBeta_tofile = zeros(n_priorBeta, 3);
            oriAlpha_tofile = zeros(n_priorBeta, n_colonies_max, 3);
            
            % specify the crystalSymmtry (CS) for parent (beta prior phase). The size
            % of Ti BCC is 3.3 by 3.3 by 3.3
            csBeta = crystalSymmetry('432',[3.32 3.32 3.32],'mineral','Ti64 (beta)');
            % specimen symmetry (which is no symmetry, or 'triclinic')
            ss = specimenSymmetry('1');
            % Similarly, specify the CS for child (alpha colonies)
            csAlpha = crystalSymmetry('622',[2.95 2.95 4.68],'mineral','Ti64 (alpha)');
            
            % generate orientations from input euler angles, the format being accepted by Neper
            if texture_strength == 0 % no texture
                if oriBeta_inp==0
                    oriBeta = project2FundamentalRegion(orientation.rand(n_priorBeta, csBeta));
                else
                    oriBeta = project2FundamentalRegion(orientation.byEuler(oriBeta_inp*degree, csBeta));
                end
            else % after 'else' are not being used
                % create a mod orientation for cube texture (001)[100]
                mod=orientation.byEuler(0,0,0,csBeta,ss);
                % create the corresponding ODF
                odf = unimodalODF(mod,'halfwidth',texture_strength*degree);
                oriBeta = discreteSample(odf,n_priorBeta);
            end
            % plot(oriBeta,'MarkerColor','b','MarkerSize',5);
            for beta = 1:n_priorBeta
                oriBeta_tofile(beta,:) = [oriBeta(beta).phi1/degree, oriBeta(beta).Phi/degree, oriBeta(beta).phi2/degree];
            end
            % Find the rotation relations (misorientations) to translate beta to alpha, according to the
            % Burgers OR
            beta2alpha = orientation.Burgers(csBeta,csAlpha);
            
            % Find all possible variants resulting from the Burgers OR. This is an
            % intermedia step, the actul beta orientation is just the onr indicated
            % by oriBeta.
            oriBetaSym = oriBeta.symmetrise;
            % Discard the last 12 variants becasue they are identical to the first 12.
            % In reality only 12 variants exists for Ti beta->alpha
            oriBetaSym = oriBetaSym(1:12,:);
            
            for beta = 1:n_priorBeta
                n_colonies_tofile(beta) = randi([n_colonies_min,n_colonies_max]);
                for colony = 1:n_colonies_tofile(beta)
                    %num_sym = randi(12); % randomly pick one variant
                    num_sym = 1; % pick the first variant
                    oriColonyBeta = oriBetaSym(num_sym, beta);
                    oriColonyAlpha_nonfund = oriColonyBeta * inv(beta2alpha); % calculate the orientation for this colony
                    oriColonyAlpha = project2FundamentalRegion(oriColonyAlpha_nonfund);
                    %organize the values to be wrtten in file. Covert radian to
                    %degrees, becasue Neper uses degrees as Euler-Bunge angles
                    %oriBeta_tofile(beta, colony,:) = [oriBeta_Bunge.phi1/degree,
                    %oriBeta_Bunge.Phi/degree, oriBeta_Bunge.phi2/degree]; %discarded. the 12 beta ori generated here are not necessary
                    oriAlpha_tofile(beta, colony,:) = [oriColonyAlpha.phi1/degree, oriColonyAlpha.Phi/degree, oriColonyAlpha.phi2/degree];
                    % plot(oriColonyAlpha,'MarkerColor','c','MarkerSize',5);
                end
            end

            % Write the results to files, angles in degree
            
            % Write number of colonies to file 'colonies'
            colonies_ID = fopen(string(path)+string(ms_ID)+'_colonies','w');
            
            priorBeta_ID = linspace(1,n_priorBeta,n_priorBeta);
            fprintf(colonies_ID,'%i %i\n',[priorBeta_ID;n_colonies_tofile']);
            
            fclose(colonies_ID);
            
            % Write file 'ori' and 'lamwidth'
            scale2_ori_ID = fopen(string(path)+string(ms_ID)+'_scale2_ori','w');
            %scale3_ori_ID = fopen(string(path)+string(ms_ID)+'_scale3_ori_pre','w');
            lamwidth_ID = fopen(string(path)+string(ms_ID)+'_lamwidth','w');

            for beta = 1:n_priorBeta
                %wirte one line in 'scale2_ori'
                fprintf(scale2_ori_ID,'%i file(%i_cell%i,des=euler-bunge)\n',[beta,ms_ID,beta]);
                % write cell file
                scale2_cell_ID = fopen(string(path)+string(ms_ID)+'_cell'+string(beta),'w');
                for i = 1:n_colonies_max
                    % print cell euler angles in each alphams_ID colony (cell).
                    fprintf(scale2_cell_ID,'%f ',oriAlpha_tofile(beta,:));
                    fprintf(scale2_cell_ID,'\n');
                end
                fclose(scale2_cell_ID);
                for colony = 1:n_colonies_tofile(beta)
                    %wirte one line in 'scale3_ori' and 'lamwidth'
                    %fprintf(scale3_ori_ID,'%i::%i file(%i_cell%i_%i,des=euler-bunge)\n',[beta,colony,ms_ID,beta,colony]);
                    % only r_lameller portion of alpha colonies have
                    % lamellar
                    islam = rand();
                    if islam <= r_lameller
                        fprintf(lamwidth_ID,'%i::%i   %.2f:%.2f\n',[beta,colony,lamwidth_alpha,lamwidth_beta]);
                    else
                        fprintf(lamwidth_ID,'%i::%i   %.2f:%.2f\n',[beta,colony,lamwidth_single,lamwidth_single]);
                    end

                    %wirte cell file
                    scale3_cell_ID = fopen(string(path)+string(ms_ID)+'_cell'+string(beta)+'_'+string(colony),'w');
                    % Write the aplha and beta orientation into a file for
                    % each grain
                    for i = 1:1
                        % print cell euler angles in each alpha colony (cell).
                        fprintf(scale3_cell_ID,'%f ',oriAlpha_tofile(beta, colony,:));
                        fprintf(scale3_cell_ID,'\n');
                        fprintf(scale3_cell_ID,'%f ',oriBeta_tofile(beta,:));
                        fprintf(scale3_cell_ID,'\n');
                    end
                    fclose(scale3_cell_ID);
                end
            end            
            fclose(scale2_ori_ID);
            %fclose(scale3_ori_ID);
            fclose(lamwidth_ID);
        end
    end
end