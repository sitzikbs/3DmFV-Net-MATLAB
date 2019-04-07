clear all
close all
clc

%% Initialize variables
trainset_path = '../data/ModelNet40/train/'; 
testset_path  = '../data/ModelNet40/test/';
log_dir = './log/';
if ~exist(log_dir,'dir')
    mkdir(log_dir);
end

%GMM variables
n_gaussians = 8;
n_points = 2048;
variance = (1/n_gaussians)^2;
normalize = true;
flatten = false;
inputSize = [n_gaussians, n_gaussians, n_gaussians, 20];
%Training variables
numClasses = 40;
max_epoch = 300;
augmentations = [false, true, true, true, false]; %rotate, scale, translation, jitter, outliers

MiniBatchSize = 128;
ExecutionEnvironment = 'gpu';
optimizer = 'adam';
InitialLearnRate = 0.001;
LearnRateSchedule = 'piecewise';
LearnRateDropPeriod = 15;
LearnRateDropFactor = 0.7;
DispatchInBackground = true;
CheckpointPath = [log_dir,'g',num2str(n_gaussians),'_n',num2str(n_points),'/'];
if ~exist(CheckpointPath, 'dir')
    mkdir(CheckpointPath)
end

%% set up the data 
[GMM] = get_3d_grid_gmm(n_gaussians, variance);
[train_pc_ds] = pc_3dmfv_data_store(trainset_path, n_points, GMM, normalize, flatten, true, augmentations);
[test_pc_ds] = pc_3dmfv_data_store(testset_path, n_points, GMM, normalize, flatten, false, augmentations);
num_train_examples = length(train_pc_ds.Files);
ValidationFrequency = uint64(5*num_train_examples/MiniBatchSize); %validate every 5 epochs
%fv_train = readimage(train_pc_ds,1);
%disp('DONE');

%% set up the network and train
 lgraph = net_3DmFV(inputSize, numClasses);
 
 options = trainingOptions(optimizer, ...
    'MaxEpochs',max_epoch, ...
    'ValidationData',test_pc_ds, ...
    'ValidationFrequency',ValidationFrequency, ...
    'Verbose',false, ...
    'MiniBatchSize', MiniBatchSize,...
    'ExecutionEnvironment',ExecutionEnvironment,...
    'InitialLearnRate', InitialLearnRate,...
    'LearnRateSchedule', LearnRateSchedule,...
    'LearnRateDropPeriod', LearnRateDropPeriod,...
    'LearnRateDropFactor', LearnRateDropFactor,...
    'DispatchInBackground',DispatchInBackground,...
    'Shuffle', 'every-epoch',...
    'CheckpointPath',CheckpointPath,...
    'Plots','training-progress');

net = trainNetwork(train_pc_ds, lgraph, options);
save([CheckpointPath, '/3DmFV_Net.mat'],'net', 'GMM', 'options', 'lgraph', 'augmentations', 'normalize', 'flatten', 'n_points'); % save the trained model and training variables
%% test the network performance
YPred = classify(net, test_pc_ds);
YValidation = test_pc_ds.Labels;
accuracy = mean(YPred == YValidation)