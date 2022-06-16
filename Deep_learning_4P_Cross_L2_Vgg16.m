clear;clc;close all;
addpath ( genpath ( 'Files mat' ) );
load('data.mat')
load('3DDmosRelease.mat');

load('norm_dL1.mat')
load Cyclopean_L1.mat

load('norm_dL2.mat')
load Cyclopean_L2.mat
% load ind_train
% load ind_test

Patch_1_L1 = [];Patch_2_L1 = [];Patch_3_L1 = [];Patch_4_L1 = [];
Patch_1_L2 = [];Patch_2_L2 = [];Patch_3_L2 = [];Patch_4_L2 = [];

Patch_1_L1(:,:,:,:) = Cyclopean_L1(1:180,1:320,:,:); % First patch
Patch_2_L1(:,:,:,:) = Cyclopean_L1(1:180,321:640,:,:); % Second patch
Patch_3_L1(:,:,:,:) = Cyclopean_L1(181:360,1:320,:,:); % Third patch
Patch_4_L1(:,:,:,:) = Cyclopean_L1(181:360,321:640,:,:); % Fourth patch

Patch_1_L2(:,:,:,:) = Cyclopean_L2(1:180,1:320,:,:); % First patch
Patch_2_L2(:,:,:,:) = Cyclopean_L2(1:180,321:640,:,:); % Second patch
Patch_3_L2(:,:,:,:) = Cyclopean_L2(181:360,1:320,:,:); % Third patch
Patch_4_L2(:,:,:,:) = Cyclopean_L2(181:360,321:640,:,:); % Fourth patch

repeat = 1; Average_R{1} =  [0 0 0 0];
for R=1:repeat
    
    % trainImages(:,:,1,:) = Cyclopean_L1(1:180,1:320,:);% First patch
    % trainImages(:,:,1,366:730) = Cyclopean_L1(1:180,321:640,:); % Second patch
    % trainImages(:,:,1,731:1095) = Cyclopean_L1(181:360,1:320,:); % Third patch
    % trainImages(:,:,1,1096:1460) = Cyclopean_L1(181:360,321:640,:); % Fourth patch
    
    %% Initialization
    N_F = 128; % Number of Features to be extracted from each patch
    Epoch = 100; %320
    M_B = 32; %60

    
   % T_input = cat(4,Patch_1_L2,Patch_2_L2,Patch_3_L2,Patch_4_L2);
   % T_output = cat(1,norm_dL2,norm_dL2,norm_dL2,norm_dL2);
     T_input = Patch_1_L2;
     T_output = norm_dL2;
    %layersTransfer = net_all.Layers(1:end);
    
    TEST_output = []; Valid = []; Score_dmos = [];
    All_NET= []; TEST_NET = [];
    
    %% Train the network for k times (Train on 80% and test on the rest 20%)
    % Transfer Learning use  : layersTransfer = alexnet.Layers(1:end-3);
       net = []; net_help = vgg16;
         
            layers = [
                imageInputLayer([180 320 3],"Name","imageinput")
                net_help.Layers(2:end-9)
               
                
                fullyConnectedLayer(N_F,"Name","fc_1")
                reluLayer("Name","relu4")
                
                fullyConnectedLayer(10,"Name","fc_2")
                reluLayer("Name","relu5")
                
                fullyConnectedLayer(1,"Name","fc_3")
                
                regressionLayer("Name","regressionoutput")];
            
            %         'Plots','training-progress'
            %         'Shuffle','every-epoch'
            
            options = trainingOptions('sgdm',...
                'LearnRateSchedule', 'piecewise',...
                'LearnRateDropFactor', 0.9,...
                'LearnRateDropPeriod', 15,...
                'L2Regularization', 0.01,...
                'MiniBatchSize',M_B,...
                'MaxEpochs',Epoch, ...
                'Shuffle','every-epoch',...
                'InitialLearnRate',1e-2);
    
        
        net = trainNetwork(T_input,T_output,layers,options)
        %net.Layers
%       Models{i} = net; % save the model
        
        layer = 'fc_1';
        
        TF_Matrix_in_L2_1 = activations(net,Patch_1_L2,layer,'OutputAs','rows'); %feature extraction 80%           
        TF_Matrix_in_L1_1 = activations(net,Patch_1_L1,layer,'OutputAs','rows');
        
        TEST_NET = predict(net,Patch_1_L1); %compute and stock results
           
    
    
    %% Evaluate the Net_Model
    
 
    disp('Results of Deep Patch 1 trained on LIVE II tested on LIVE I>>>');
    SB1 = [dmos];               % Subjective Score
    OB1 = TEST_NET;                    % Objective Score
    %figure,
    [Srocc,Kroo, cc,rmse] = logistic_cc(double(OB1),double(SB1))
    
    %% Train the SVR with the extracted features Patch 1
    
    TEST_SVR=[]; All_SVR = [];           
        
        Mdln = fitrsvm(TF_Matrix_in_L2_1,T_output,'verbose',0,'KernelFunction','Gaussian','KernelScale','auto');
        TEST_SVR = Mdln.predict(TF_Matrix_in_L1_1);
        
   
    
    %% Evaluate the SVR_Model Patch 1
   
    disp('Results of SVR-Features Patch 1 trained on LIVE II tested on LIVE I>>>');
    SB2 = [dmos];                 % Subjective Score
    OB2 = TEST_SVR;                      % Objective Score
    %figure,
    [Srocc,Kroo, cc,rmse] = logistic_cc(double(OB2),double(SB2))
    
    %% Second patch   
    
    %% Initialization
    T_input = Patch_2_L2 ;
    T_output = [norm_dL2];
    %layersTransfer = net_all.Layers(1:end);
    
      TEST_output = []; Valid = []; Score_dmos = [];
    All_NET= []; TEST_NET = [];
    
    %% Train the network for k times (Train on 80% and test on the rest 20%)
    % Transfer Learning use  : layersTransfer = alexnet.Layers(1:end-3);
    
        net = []; net_help = vgg16;
         
            layers = [
                imageInputLayer([180 320 3],"Name","imageinput")
                net_help.Layers(2:end-9)
               
                
                fullyConnectedLayer(N_F,"Name","fc_1")
                reluLayer("Name","relu4")
                
                fullyConnectedLayer(10,"Name","fc_2")
                reluLayer("Name","relu5")
                
                fullyConnectedLayer(1,"Name","fc_3")
                
                regressionLayer("Name","regressionoutput")];
            
            %         'Plots','training-progress'
            %         'Shuffle','every-epoch'
            
            options = trainingOptions('sgdm',...
                'LearnRateSchedule', 'piecewise',...
                'LearnRateDropFactor', 0.9,...
                'LearnRateDropPeriod', 15,...
                'L2Regularization', 0.01,...
                'MiniBatchSize',M_B,...
                'MaxEpochs',Epoch, ...
                'Shuffle','every-epoch',...
                'InitialLearnRate',1e-2);
    
        
        net = trainNetwork(T_input,T_output,layers,options)
        %net.Layers
%       Models{i} = net; % save the model
        
        layer = 'fc_1';
        
        TF_Matrix_in_L2_2 = activations(net,Patch_2_L2,layer,'OutputAs','rows'); %feature extraction 80%           
        TF_Matrix_in_L1_2 = activations(net,Patch_2_L1,layer,'OutputAs','rows');
        
        TEST_NET = predict(net,Patch_2_L1); %compute and stock results
           
    
    
    %% Evaluate the Net_Model
    
 
    disp('Results of Deep Patch 2 trained on LIVE II tested on LIVE I>>>');
    SB1 = [dmos];               % Subjective Score
    OB1 = TEST_NET;                    % Objective Score
    %figure,
    [Srocc,Kroo, cc,rmse] = logistic_cc(double(OB1),double(SB1))
    
    %% Train the SVR with the extracted features Patch 1
    
    TEST_SVR=[]; All_SVR = [];           
        
        Mdln = fitrsvm(TF_Matrix_in_L2_2,T_output,'verbose',0,'KernelFunction','Gaussian','KernelScale','auto');
        TEST_SVR = Mdln.predict(TF_Matrix_in_L1_2);
        
   
    
    %% Evaluate the SVR_Model Patch 1
   
    disp('Results of SVR-Features Patch 2 trained on LIVE II tested on LIVE I>>>');
    SB2 = [dmos];                 % Subjective Score
    OB2 = TEST_SVR;                      % Objective Score
    %figure,
    [Srocc,Kroo, cc,rmse] = logistic_cc(double(OB2),double(SB2))
 
    
    %% 3rd Patch
    
    %% Initialization
    T_input = Patch_3_L2 ;
    T_output = [norm_dL2];
    %layersTransfer = net_all.Layers(1:end);
 
      TEST_output = []; Valid = []; Score_dmos = [];
    All_NET= []; TEST_NET = [];
    
    %% Train the network for k times (Train on 80% and test on the rest 20%)
    % Transfer Learning use  : layersTransfer = alexnet.Layers(1:end-3);
      net = []; net_help = vgg16;
         
            layers = [
                imageInputLayer([180 320 3],"Name","imageinput")
                net_help.Layers(2:end-9)
               
                
                fullyConnectedLayer(N_F,"Name","fc_1")
                reluLayer("Name","relu4")
                
                fullyConnectedLayer(10,"Name","fc_2")
                reluLayer("Name","relu5")
                
                fullyConnectedLayer(1,"Name","fc_3")
                
                regressionLayer("Name","regressionoutput")];
            
            %         'Plots','training-progress'
            %         'Shuffle','every-epoch'
            
            options = trainingOptions('sgdm',...
                'LearnRateSchedule', 'piecewise',...
                'LearnRateDropFactor', 0.9,...
                'LearnRateDropPeriod', 15,...
                'L2Regularization', 0.01,...
                'MiniBatchSize',M_B,...
                'MaxEpochs',Epoch, ...
                'Shuffle','every-epoch',...
                'InitialLearnRate',1e-2);
    
        net = trainNetwork(T_input,T_output,layers,options)
        %net.Layers
%       Models{i} = net; % save the model
        
        layer = 'fc_1';
        
        TF_Matrix_in_L2_3 = activations(net,Patch_3_L2,layer,'OutputAs','rows'); %feature extraction 80%           
        TF_Matrix_in_L1_3 = activations(net,Patch_3_L1,layer,'OutputAs','rows');
        
        TEST_NET = predict(net,Patch_3_L1); %compute and stock results
           
    
    
    %% Evaluate the Net_Model
    
 
    disp('Results of Deep Patch 3 trained on LIVE II tested on LIVE I>>>');
    SB1 = [dmos];               % Subjective Score
    OB1 = TEST_NET;                    % Objective Score
    %figure,
    [Srocc,Kroo, cc,rmse] = logistic_cc(double(OB1),double(SB1))
    
    %% Train the SVR with the extracted features Patch 1
    
    TEST_SVR=[]; All_SVR = [];           
        
        Mdln = fitrsvm(TF_Matrix_in_L2_3,T_output,'verbose',0,'KernelFunction','Gaussian','KernelScale','auto');
        TEST_SVR = Mdln.predict(TF_Matrix_in_L1_3);
        
   
    
    %% Evaluate the SVR_Model Patch 1
   
    disp('Results of SVR-Features Patch 3 trained on LIVE II tested on LIVE I>>>');
    SB2 = [dmos];                 % Subjective Score
    OB2 = TEST_SVR;                      % Objective Score
    %figure,
    [Srocc,Kroo, cc,rmse] = logistic_cc(double(OB2),double(SB2))
    
    
    %% 4th Patch
    %% Initialization
    T_input = Patch_4_L2 ;
    T_output = [norm_dL2];
    %layersTransfer = net_all.Layers(1:end);
    
   TEST_output = []; Valid = []; Score_dmos = [];
    All_NET= []; TEST_NET = [];
    
    %% Train the network for k times (Train on 80% and test on the rest 20%)
    % Transfer Learning use  : layersTransfer = alexnet.Layers(1:end-3);
    
        
        net = []; net_help = vgg16;
         
            layers = [
                imageInputLayer([180 320 3],"Name","imageinput")
                net_help.Layers(2:end-9)
               
                
                fullyConnectedLayer(N_F,"Name","fc_1")
                reluLayer("Name","relu4")
                
                fullyConnectedLayer(10,"Name","fc_2")
                reluLayer("Name","relu5")
                
                fullyConnectedLayer(1,"Name","fc_3")
                
                regressionLayer("Name","regressionoutput")];
            
            %         'Plots','training-progress'
            %         'Shuffle','every-epoch'
            
            options = trainingOptions('sgdm',...
                'LearnRateSchedule', 'piecewise',...
                'LearnRateDropFactor', 0.9,...
                'L2Regularization', 0.01,...
                'LearnRateDropPeriod', 15,...
                'MiniBatchSize',M_B,...
                'MaxEpochs',Epoch, ...
                'Shuffle','every-epoch',...
                'InitialLearnRate',1e-2);
    
        
        net = trainNetwork(T_input,T_output,layers,options)
        %net.Layers
%       Models{i} = net; % save the model
        
        layer = 'fc_1';
        
        TF_Matrix_in_L2_4 = activations(net,Patch_4_L2,layer,'OutputAs','rows'); %feature extraction 80%           
        TF_Matrix_in_L1_4 = activations(net,Patch_4_L1,layer,'OutputAs','rows');
        
        TEST_NET = predict(net,Patch_4_L1); %compute and stock results
           
    
    
    %% Evaluate the Net_Model
    
 
    disp('Results of Deep Patch 4 trained on LIVE II tested on LIVE I>>>');
    SB1 = [dmos];               % Subjective Score
    OB1 = TEST_NET;                    % Objective Score
    %figure,
    [Srocc,Kroo, cc,rmse] = logistic_cc(double(OB1),double(SB1))
    
    %% Train the SVR with the extracted features Patch 1
    
    TEST_SVR=[]; All_SVR = [];           
        
        Mdln = fitrsvm(TF_Matrix_in_L2_4,T_output,'verbose',0,'KernelFunction','Gaussian','KernelScale','auto');
        TEST_SVR = Mdln.predict(TF_Matrix_in_L1_4);
        
   
    
    %% Evaluate the SVR_Model Patch 1
   
    disp('Results of SVR-Features Patch 4 trained on LIVE II tested on LIVE I>>>');
    SB2 = [dmos];                 % Subjective Score
    OB2 = TEST_SVR;                      % Objective Score
    %figure,
    [Srocc,Kroo, cc,rmse] = logistic_cc(double(OB2),double(SB2))
    
    
    %% Train the SVR with the all extracted features all Patches
   
        TF_Matrix_in_all = [TF_Matrix_in_L2_1 TF_Matrix_in_L2_2 TF_Matrix_in_L2_3 TF_Matrix_in_L2_4];
        TF_Matrix_out_all = [TF_Matrix_in_L1_1 TF_Matrix_in_L1_2 TF_Matrix_in_L1_3 TF_Matrix_in_L1_4];
   
        TEST_SVR=[]; All_SVR = [];
    

        
        Mdln = fitrsvm(TF_Matrix_in_all, norm_dL2,'verbose',0,'KernelFunction','Gaussian','KernelScale','auto');
        TEST_SVR = Mdln.predict(TF_Matrix_out_all);
        All_SVR = [All_SVR; TEST_SVR];
        
        
    
    
    %% Evaluate the SVR_Model -all Patches

    disp('Results of SVR-Features All Patches trained on LIVE II tested on LIVE I>>>');
    SB2 = [dmos];                 % Subjective Score
    OB2 = All_SVR;                      % Objective Score
    %figure,
    [Srocc,Krooc, cc,rmse] = logistic_cc(double(OB2),double(SB2))
    
    Score_Stock{R} = All_SVR;
    Final_R{R}=[Srocc,Krooc, cc,rmse];
    Average_R{1} = Average_R{1} + Final_R{R};
    
end

Average_R{1} = Average_R{1} / repeat;
disp(Average_R{1})
