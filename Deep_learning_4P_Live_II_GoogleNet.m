clear;clc;close all;
addpath ( genpath ( 'Files mat' ) );
load('3DDmosRelease.mat');
load('norm_dL2.mat')
load Cyclopean_L2.mat
% load ind_train
% load ind_test

Patch_1 = [];Patch_2 = [];Patch_3 = [];Patch_4 = [];

Patch_1(:,:,:,:) = Cyclopean_L2(1:180,1:320,:,:); % First patch
Patch_2(:,:,:,:) = Cyclopean_L2(1:180,321:640,:,:); % Second patch
Patch_3(:,:,:,:) = Cyclopean_L2(181:360,1:320,:,:); % Third patch
Patch_4(:,:,:,:) = Cyclopean_L2(181:360,321:640,:,:); % Fourth patch

repeat = 1; Average_R{1} =  [0 0 0 0];
for R=1:repeat
    
    % trainImages(:,:,1,:) = Cyclopean_L1(1:180,1:320,:);% First patch
    % trainImages(:,:,1,366:730) = Cyclopean_L1(1:180,321:640,:); % Second patch
    % trainImages(:,:,1,731:1095) = Cyclopean_L1(181:360,1:320,:); % Third patch
    % trainImages(:,:,1,1096:1460) = Cyclopean_L1(181:360,321:640,:); % Fourth patch
    
    %% Initialization
    N_F = 128; % Number of Features to be extracted from each patch
    Epoch = 50; %320
    M_B = 16; %60
    
    T_input = Patch_1;
    T_output = [norm_dL2];
    
    %layersTransfer = net_all.Layers(1:end);
    
    TEST_output = []; Valid = []; Score_dmos = [];
    All_NET= []; TEST_NET = [];
    k = 5;
    %% Divide dataset to 80%-20% (Non-overlapped)
    
    cv = cvpartition(length(T_output), 'kfold',k);   

    for i=1:k
        
        trainIdxs{i} = find(training(cv,i));  %trainIdxs{i} = find(ind_train(:,i)==1);
        testIdxs{i}  = find(test(cv,i));      %testIdxs{i}  = find(ind_test(:,i)==1);
        
        trainMatrix_IN{i} = [T_input(:,:,:,trainIdxs{i})]; %Inout and output for the network 80%
        trainMatrix_OUT{i} = [T_output(trainIdxs{i})];
        
        testMatrix_IN{i} = [T_input(:,:,:,testIdxs{i})];  %Input for Testing 20%
        testMatrix_OUT{i} = [T_output(testIdxs{i})];
        Subjectiv_S{i} = [Dmos(testIdxs{i})];
        
    end
    
    %% Train the network for k times (Train on 80% and test on the rest 20%)
    
       net = googlenet;
        lgraph = layerGraph(net);
        
        %   REMOVE LAYERS
        lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
        
        
        %   MODEL TO ADD
        newLayers = [
            fullyConnectedLayer(N_F,"Name","fc_1")
            reluLayer("Name","relu4")
            
            fullyConnectedLayer(10,"Name","fc_11")
            %reluLayer("Name","relu5")
            
            fullyConnectedLayer(1,"Name","fc_3")
            
            regressionLayer("Name","regressionoutput")];
        lgraph = addLayers(lgraph,newLayers);
        
        %   CHANGE THE INPUT SIZE
        lgraph = removeLayers(lgraph, {'data'})
        LLL=imageInputLayer([180 320 3],'Name','input');
        lgraph = addLayers(lgraph,LLL);
        lgraph = connectLayers(lgraph,'input','conv1-7x7_s2');
        
        %   CHANGE THE LAST AVERAGE POOLING
        lgraph = removeLayers(lgraph, {'pool5-7x7_s1'})
        %AVP = averagePooling2dLayer(1,'Name','avg1');
        %lgraph = addLayers(lgraph,AVP);
        lgraph = connectLayers(lgraph,'inception_5b-output','pool5-drop_7x7_s1');
        lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc_1');
        
        addpath /home/dian/Matlab/examples/nnet/main
        layers = lgraph.Layers;
        connections = lgraph.Connections;
        lgraph = createLgraphUsingConnections(layers,connections);
        
        %         'Plots','training-progress'
        %         'Shuffle','every-epoch'
        
        options = trainingOptions('sgdm',...
            'LearnRateSchedule', 'piecewise',...
            'LearnRateDropFactor', 0.9,...
            'LearnRateDropPeriod', 10,...
            'MiniBatchSize',M_B,...
            'MaxEpochs',Epoch, ...
            'Shuffle','every-epoch',...
            'InitialLearnRate',1e-2);
       
    
    
    % Transfer Learning use  : layersTransfer = alexnet.Layers(1:end-3);
    for i = 1:k
        
      
        
        
        net = trainNetwork(trainMatrix_IN{i},trainMatrix_OUT{i},lgraph,options)
        %net.Layers
        Models{1,i} = net; % save the model
        
        layer = 'fc_1';
        TF_Matrix_in_1{i} = activations(net,trainMatrix_IN{i},layer,'OutputAs','rows'); %feature extraction 80%
        TF_Matrix_out_1{i} = activations(net,testMatrix_IN{i},layer,'OutputAs','rows');%feature extraction 20%
        
        TEST_NET = predict(net,testMatrix_IN{i}); %compute and stock results
        All_NET = [All_NET; TEST_NET];
        Score_dmos = [Score_dmos; Subjectiv_S{i}]; %re-order the dmos score
    end
    
    
    %% Evaluate the Net_Model
    
    Score_1 = All_NET;
    disp('Results of Deep Patch 1 LIVE II>>>');
    SB1 = [Score_dmos];               % Subjective Score
    OB1 = Score_1;                    % Objective Score
    %figure,
    [Srocc,Krooc, cc,rmse] = logistic_cc(double(OB1),double(SB1))
    
    %% Train the SVR with the extracted features Patch 1
    
    Score_dmos = []; TEST_SVR=[]; All_SVR = [];
    
    for i = 1:k
        
        
        Mdln = fitrsvm(TF_Matrix_in_1{i}, trainMatrix_OUT{i},'verbose',0,'KernelFunction','Gaussian','KernelScale','auto');
        TEST_SVR = Mdln.predict(TF_Matrix_out_1{i});
        All_SVR = [All_SVR; TEST_SVR];
        Score_dmos = [Score_dmos; Subjectiv_S{i}]; %re-order the dmos score
        
    end
    
    %% Evaluate the SVR_Model Patch 1
    Score_2 = All_SVR;
    disp('Results of SVR-Features Patch 1 LIVE II>>>');
    SB2 = [Score_dmos];                 % Subjective Score
    OB2 = Score_2;                      % Objective Score
    %figure,
    [Srocc,Krooc, cc,rmse] = logistic_cc(double(OB2),double(SB2))
    
    %% Second patch
    
    
    %% Initialization
    T_input = Patch_2 ;
    T_output = [norm_dL2];
    %layersTransfer = net_all.Layers(1:end);
    
    TEST_output = []; Valid = []; Score_dmos = [];
    All_NET= []; TEST_NET = [];
    
    %% Divide dataset to 80%-20% (Non-overlapped)
      
    for i=1:k
        
%         trainIdxs{i} = find(ind_train(:,i)==1);
%         testIdxs{i}  = find(ind_test(:,i)==1);
        
        trainMatrix_IN{i} = [T_input(:,:,:,trainIdxs{i})]; %Inout and output for the network 80%
        trainMatrix_OUT{i} = [T_output(trainIdxs{i})];
        
        testMatrix_IN{i} = [T_input(:,:,:,testIdxs{i})];  %Input for Testing 20%
        testMatrix_OUT{i} = [T_output(testIdxs{i})];
        Subjectiv_S{i} = [Dmos(testIdxs{i})];
        
    end
    
    %% Train the network for k times (Train on 80% and test on the rest 20%)
    % Transfer Learning use  : layersTransfer = alexnet.Layers(1:end-3);
     
        %         'Plots','training-progress'
        %         'Shuffle','every-epoch'
            
    
    for i = 1:k
        
     
           
        net = trainNetwork(trainMatrix_IN{i},trainMatrix_OUT{i},lgraph,options)
        %net.Layers
        Models{2,i} = net; % save the model
        
        layer = 'fc_1';
        TF_Matrix_in_2{i} = activations(net,trainMatrix_IN{i},layer,'OutputAs','rows'); %feature extraction 80%
        TF_Matrix_out_2{i} = activations(net,testMatrix_IN{i},layer,'OutputAs','rows');%feature extraction 20%
        
        TEST_NET = predict(net,testMatrix_IN{i}); %compute and stock results
        All_NET = [All_NET; TEST_NET];
        Score_dmos = [Score_dmos; Subjectiv_S{i}]; %re-order the dmos score
    end
    %% Evaluate the Net_Model
    
    Score_1 = All_NET;
    disp('Results of Deep Patch 2 LIVE II>>>');
    SB1 = [Score_dmos];               % Subjective Score
    OB1 = Score_1;                    % Objective Score
    %figure,
    [Srocc,Krooc, cc,rmse] = logistic_cc(double(OB1),double(SB1))
    
    
    %% Train the SVR with the extracted features Patch 2
    
    Score_dmos = []; TEST_SVR=[]; All_SVR = [];
    
    for i = 1:k
        
        
        Mdln = fitrsvm(TF_Matrix_in_2{i}, trainMatrix_OUT{i},'verbose',0,'KernelFunction','Gaussian','KernelScale','auto');
        TEST_SVR = Mdln.predict(TF_Matrix_out_2{i});
        All_SVR = [All_SVR; TEST_SVR];
        Score_dmos = [Score_dmos; Subjectiv_S{i}]; %re-order the dmos score
        
    end
    
    %% Evaluate the SVR_Model Patch 2
    Score_2 = All_SVR;
    disp('Results of SVR-Features Patch 2 LIVE II>>>');
    SB2 = [Score_dmos];                 % Subjective Score
    OB2 = Score_2;                      % Objective Score
    %figure,
    [Srocc,Krooc, cc,rmse] = logistic_cc(double(OB2),double(SB2))
    
    %% 3rd Patch
    
    %% Initialization
    T_input = Patch_3 ;
    T_output = [norm_dL2];
    %layersTransfer = net_all.Layers(1:end);
    
    TEST_output = []; Valid = []; Score_dmos = [];
    All_NET= []; TEST_NET = [];
    
    %% Divide dataset to 80%-20% (Non-overlapped)
    
    for i=1:k
        
%         trainIdxs{i} = find(ind_train(:,i)==1);
%         testIdxs{i}  = find(ind_test(:,i)==1);
        
        trainMatrix_IN{i} = [T_input(:,:,:,trainIdxs{i})]; %Inout and output for the network 80%
        trainMatrix_OUT{i} = [T_output(trainIdxs{i})];
        
        testMatrix_IN{i} = [T_input(:,:,:,testIdxs{i})];  %Input for Testing 20%
        testMatrix_OUT{i} = [T_output(testIdxs{i})];
        Subjectiv_S{i} = [Dmos(testIdxs{i})];
        
    end
    
    %% Train the network for k times (Train on 80% and test on the rest 20%)
    % Transfer Learning use  : layersTransfer = alexnet.Layers(1:end-3);
    
       
    for i = 1:k
        
        net = trainNetwork(trainMatrix_IN{i},trainMatrix_OUT{i},lgraph,options)
        %net.Layers
        Models{3,i} = net; % save the model
        
        layer = 'fc_1';
        TF_Matrix_in_3{i} = activations(net,trainMatrix_IN{i},layer,'OutputAs','rows'); %feature extraction 80%
        TF_Matrix_out_3{i} = activations(net,testMatrix_IN{i},layer,'OutputAs','rows');%feature extraction 20%
        
        TEST_NET = predict(net,testMatrix_IN{i}); %compute and stock results
        All_NET = [All_NET; TEST_NET];
        Score_dmos = [Score_dmos; Subjectiv_S{i}]; %re-order the dmos score
    end
    %% Evaluate the Net_Model
    
    Score_1 = All_NET;
    disp('Results of Deep Patch 3 LIVE II>>>');
    SB1 = [Score_dmos];               % Subjective Score
    OB1 = Score_1;                    % Objective Score
    %figure,
    [Srocc,Krooc, cc,rmse] = logistic_cc(double(OB1),double(SB1))
    
    %% Train the SVR with the extracted features Patch 3
    
    Score_dmos = []; TEST_SVR=[]; All_SVR = [];
    
    for i = 1:k
        
        
        Mdln = fitrsvm(TF_Matrix_in_3{i}, trainMatrix_OUT{i},'verbose',0,'KernelFunction','Gaussian','KernelScale','auto');
        TEST_SVR = Mdln.predict(TF_Matrix_out_3{i});
        All_SVR = [All_SVR; TEST_SVR];
        Score_dmos = [Score_dmos; Subjectiv_S{i}]; %re-order the dmos score
        
    end
    
    %% Evaluate the SVR_Model Patch 3
    Score_2 = All_SVR;
    disp('Results of SVR-Features Patch 3 LIVE II>>>');
    SB2 = [Score_dmos];                 % Subjective Score
    OB2 = Score_2;                      % Objective Score
    %figure,
    [Srocc,Krooc, cc,rmse] = logistic_cc(double(OB2),double(SB2))
    
    %% 4th Patch
    %% Initialization
    T_input = Patch_4 ;
    T_output = [norm_dL2];
    %layersTransfer = net_all.Layers(1:end);
    
    TEST_output = []; Valid = []; Score_dmos = [];
    All_NET= []; TEST_NET = [];
    
    %% Divide dataset to 80%-20% (Non-overlapped)
    
    
    for i=1:k
        
%         trainIdxs{i} = find(ind_train(:,i)==1);
%         testIdxs{i}  = find(ind_test(:,i)==1);
        
        trainMatrix_IN{i} = [T_input(:,:,:,trainIdxs{i})]; %Inout and output for the network 80%
        trainMatrix_OUT{i} = [T_output(trainIdxs{i})];
        
        testMatrix_IN{i} = [T_input(:,:,:,testIdxs{i})];  %Input for Testing 20%
        testMatrix_OUT{i} = [T_output(testIdxs{i})];
        Subjectiv_S{i} = [Dmos(testIdxs{i})];
        
    end
    
    %% Train the network for k times (Train on 80% and test on the rest 20%)
    % Transfer Learning use  : layersTransfer = alexnet.Layers(1:end-3);
    
    for i = 1:k
      
        
        net = trainNetwork(trainMatrix_IN{i},trainMatrix_OUT{i},lgraph,options)
        %net.Layers
        Models{4,i} = net; % save the model
        
        layer = 'fc_1';
        TF_Matrix_in_4{i} = activations(net,trainMatrix_IN{i},layer,'OutputAs','rows'); %feature extraction 80%
        TF_Matrix_out_4{i} = activations(net,testMatrix_IN{i},layer,'OutputAs','rows');%feature extraction 20%
        
        TEST_NET = predict(net,testMatrix_IN{i}); %compute and stock results
        All_NET = [All_NET; TEST_NET];
        Score_dmos = [Score_dmos; Subjectiv_S{i}]; %re-order the dmos score
    end
    %% Evaluate the Net_Model
    
    Score_1 = All_NET;
    disp('Results of Deep Patch 4 LIVE II>>>');
    SB1 = [Score_dmos];               % Subjective Score
    OB1 = Score_1;                    % Objective Score
    %figure,
    [Srocc,Krooc, cc,rmse] = logistic_cc(double(OB1),double(SB1))
    
    %% Train the SVR with the extracted features Patch 4
    
    Score_dmos = []; TEST_SVR=[]; All_SVR = [];
    
    for i = 1:k
        
        
        Mdln = fitrsvm(TF_Matrix_in_4{i}, trainMatrix_OUT{i},'verbose',0,'KernelFunction','Gaussian','KernelScale','auto');
        TEST_SVR = Mdln.predict(TF_Matrix_out_4{i});
        All_SVR = [All_SVR; TEST_SVR];
        Score_dmos = [Score_dmos; Subjectiv_S{i}]; %re-order the dmos score
        
    end
    
    %% Evaluate the SVR_Model Patch 4
    Score_2 = All_SVR;
    disp('Results of SVR-Features Patch 4 LIVE II>>>');
    SB2 = [Score_dmos];                 % Subjective Score
    OB2 = Score_2;                      % Objective Score
    %figure,
    [Srocc,Krooc, cc,rmse] = logistic_cc(double(OB2),double(SB2))
    
    
    %% Train the SVR with the all extracted features all Patches
    for i=1:k
        TF_Matrix_in_all{i} = [TF_Matrix_in_1{i} TF_Matrix_in_2{i} TF_Matrix_in_3{i} TF_Matrix_in_4{i}];
        TF_Matrix_out_all{i} = [TF_Matrix_out_1{i} TF_Matrix_out_2{i} TF_Matrix_out_3{i} TF_Matrix_out_4{i}];
    end
    Score_dmos = []; TEST_SVR=[]; All_SVR = []; DATA = []; DATA_C =[];
    
    for i = 1:k
        
        
        Mdln = fitrsvm(TF_Matrix_in_all{i}, trainMatrix_OUT{i},'verbose',0,'KernelFunction','Gaussian','KernelScale','auto');
        TEST_SVR = Mdln.predict(TF_Matrix_out_all{i});
        All_SVR = [All_SVR; TEST_SVR];
        Score_dmos = [Score_dmos; Subjectiv_S{i}]; %re-order the dmos score to versus the nandomly split
        
        DATA = [DATA; testIdxs{i}]; %re-order the dmos score to the first order of DMOS
    end
    
    %% Evaluate the SVR_Model -all Patches
    for j=1:360
        
       DATA_C(DATA(j)) = [All_SVR(j)]; %re-order the dmos score to the first order of DMOS
       
    end
    
   
    disp('Results of SVR-Features All Patches LIVE II>>>');
    SB2 = Dmos;                 % Subjective Score
    OB2 = DATA_C';                    % Objective Score
    %figure,
    [Srocc,Krooc, cc,rmse] = logistic_cc(double(OB2),double(SB2))
    
    Score_Stock{R} = DATA_C';
    Final_R{R}=[Srocc,Krooc, cc,rmse];
    Average_R{1} = Average_R{1} + Final_R{R};
    
end

Average_R{1} = Average_R{1} / repeat;
disp(Average_R{1})
