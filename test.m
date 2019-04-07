clear all
close all
clc

results_path = './log/g8/';
testset_path  = '/home/itzik/MatlabProjects/3DmFVNet/data/ModelNet40/test/';
load([results_path, '3DmFV_Net.mat']);

[test_pc_ds] = pc_3dmfv_data_store(testset_path, n_points, GMM, normalize, flatten, false, augmentations);

YPred = classify(net, test_pc_ds);
YValidation = test_pc_ds.Labels;
accuracy = mean(YPred == YValidation)