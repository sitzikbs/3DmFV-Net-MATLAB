function lgraph = net_3DmFV(inputSize, numClasses)
% 3DmFV-Net architecture
% Input:
%        inputSize: input size ( length 4 vector )
%       numClasses: number of classes [int scalar]
% Output:
%        lgraph: the current layer graph        
lgraph = layerGraph;
lgraph = addLayers(lgraph, image3dInputLayer(inputSize, 'Normalization', 'none', 'name','input_layer'));
[lgraph, cnn_last_layer_name] = CNN_3DmFV_Net(lgraph, inputSize);

non_linear_classifier_layers = non_linear_classifier(numClasses, 'non_linear_classifier');
lgraph = addLayers(lgraph, non_linear_classifier_layers);

lgraph = connectLayers(lgraph, cnn_last_layer_name, 'fc_1_non_linear_classifier');
% plot(lgraph)
end

function [lgraph, module_last_layer_name] = CNN_3DmFV_Net(lgraph, inputSize)
% CNN_3DmFV_Net Constructs the graph of the whole CNN part of 3DmFV-Net which is composed
% of several inception and maxpooling layer ( depends on the number of
% Gaussians). 
%Input:
%        lgraph: the current layer graph
%        inputSize: input size ( length 4 vector )
%Output : 
%       lgraph: the constructed layer graph
%       module_last_layer_name: string of the last layer name to be used in
%       follwing layers
switch inputSize(1)
    case 3
        [lgraph, module_last_layer_name] =  inception3D(64, [2, 3], lgraph, 'input_layer', 'inception_1');
        [lgraph, module_last_layer_name] =  inception3D(128, [2, 3], lgraph, module_last_layer_name, 'inception_2');
        [lgraph, module_last_layer_name] =  inception3D(256, [2, 3], lgraph, module_last_layer_name, 'inception_3');
        [lgraph, module_last_layer_name] =  inception3D(256, [2, 3], lgraph, module_last_layer_name, 'inception_4');
        [lgraph, module_last_layer_name] =  inception3D(512, [2, 3], lgraph, module_last_layer_name, 'inception_5');
    case 5
        [lgraph, module_last_layer_name] =  inception3D(64, [3, 5], lgraph, 'input_layer', 'inception_1');
        [lgraph, module_last_layer_name] =  inception3D(128, [3, 5], lgraph, module_last_layer_name, 'inception_2');
        [lgraph, module_last_layer_name] =  inception3D(256, [3, 5], lgraph, module_last_layer_name, 'inception_3');
        lgraph = addLayers(lgraph, maxPooling3dLayer(3, 'stride',[2, 2, 2], 'name', 'maxpool'));
        lgraph = connectLayers(lgraph, module_last_layer_name, 'maxpool');
        [lgraph, module_last_layer_name] =  inception3D(256, [2, 3], lgraph, 'maxpool', 'inception_4');
        [lgraph, module_last_layer_name] =  inception3D(512, [2, 3], lgraph, module_last_layer_name, 'inception_5');
    case 8
        [lgraph, module_last_layer_name] =  inception3D(64, [3, 5], lgraph, 'input_layer', 'inception_1');
        [lgraph, module_last_layer_name] =  inception3D(128, [3, 5], lgraph, module_last_layer_name, 'inception_2');
        [lgraph, module_last_layer_name] =  inception3D(256, [3, 5], lgraph, module_last_layer_name, 'inception_3');
        lgraph = addLayers(lgraph, maxPooling3dLayer(2, 'stride',[2, 2, 2], 'name', 'maxpool_1'));
        lgraph = connectLayers(lgraph, module_last_layer_name, 'maxpool_1');
        [lgraph, module_last_layer_name] =  inception3D(256, [3, 4], lgraph, 'maxpool_1', 'inception_4');
        [lgraph, module_last_layer_name] =  inception3D(512, [3, 4], lgraph, module_last_layer_name, 'inception_5');
        lgraph = addLayers(lgraph, maxPooling3dLayer(2, 'stride',[2, 2, 2], 'name', 'maxpool_2'));
        lgraph = connectLayers(lgraph, module_last_layer_name, 'maxpool_2');
        module_last_layer_name = 'maxpool_2';
    case 16
        [lgraph, module_last_layer_name] =  inception3D(64 , [4, 8], lgraph, 'input_layer', 'inception_1');
        [lgraph, module_last_layer_name] =  inception3D(128, [4, 8], lgraph, module_last_layer_name, 'inception_2');
        [lgraph, module_last_layer_name] =  inception3D(256, [4, 8], lgraph, module_last_layer_name, 'inception_3');
        lgraph = addLayers(lgraph, maxPooling3dLayer(2, 'stride',[2, 2, 2], 'name', 'maxpool_1'));
        lgraph = connectLayers(lgraph, module_last_layer_name, 'maxpool_1');
        [lgraph, module_last_layer_name] =  inception3D(256, [3, 5], lgraph, 'maxpool_1', 'inception_4');
        [lgraph, module_last_layer_name] =  inception3D(512, [3, 5], lgraph, module_last_layer_name, 'inception_5');
        lgraph = addLayers(lgraph, maxPooling3dLayer(2, 'stride',[2, 2, 2], 'name', 'maxpool_2'));
        lgraph = connectLayers(lgraph, module_last_layer_name, 'maxpool_2');
        [lgraph, module_last_layer_name] =  inception3D(512, [2, 3], lgraph, 'maxpool_2', 'inception_6');
        [lgraph, module_last_layer_name] =  inception3D(512, [2, 3], lgraph, module_last_layer_name, 'inception_7');
        lgraph = addLayers(lgraph, maxPooling3dLayer(2, 'stride',[2, 2, 2], 'name', 'maxpool_3'));
        lgraph = connectLayers(lgraph, module_last_layer_name, 'maxpool_3');
        module_last_layer_name = 'maxpool_3';
    otherwise
        error('Unsupported number of Gaussians');
end
end

function [lgraph, module_last_layer_name] = inception3D(n_filters, kernel_sizes, lgraph, input_layer, module_scope)
% Constructs the graph of the Inception inspired module. outputs the concatenation of 3D CNN outputs
% and avg pooling layers (3*n_filters output channels).
%Input:
%       n_filters: number of filters to compute
%        kernel_sizes: filter sizes (length 2 vector)
%        lgraph: the current layer graph
%       input layer: name of input layer for connection [string]
%       module_scope: unique name for module (to allow for multiple uses of the
%       module [string]
%Output : 
%        lgraph: the constructed layer graph
%       module_last_layer_name: string of the last layer name to be used in
% follwing layers
lgraph = addLayers(lgraph, [convolution3dLayer(1,n_filters, 'padding', 'same', 'Name',['cnn_1', module_scope])
    batchNormalizationLayer('Name',['bn_1' , module_scope])
    reluLayer('Name',['relu_1' , module_scope])]);
lgraph = addLayers(lgraph, [convolution3dLayer(kernel_sizes(1),int8(n_filters/2), 'padding', 'same', 'Name',['cnn_2', module_scope])
    batchNormalizationLayer('Name',['bn_2', module_scope])
    reluLayer('Name',['relu_2', module_scope])]);
lgraph = addLayers(lgraph, [convolution3dLayer(kernel_sizes(2),int8(n_filters/2), 'padding', 'same', 'Name',['cnn_3', module_scope])
    batchNormalizationLayer('Name',['bn_3', module_scope])
    reluLayer('Name',['relu_3', module_scope])]);
lgraph = addLayers(lgraph, [averagePooling3dLayer(kernel_sizes(1), 'padding', 'same', 'Name',['avg_pool', module_scope])
    convolution3dLayer(1,n_filters,  'Name',['cnn_4', module_scope])
    batchNormalizationLayer('Name',['bn_4', module_scope])
    reluLayer('Name',['relu_4', module_scope])]);

lgraph = connectLayers(lgraph, input_layer,['cnn_1', module_scope]);
lgraph = connectLayers(lgraph, input_layer,['cnn_2', module_scope]);
lgraph = connectLayers(lgraph, input_layer,['cnn_3', module_scope]);
lgraph = connectLayers(lgraph, input_layer,['avg_pool', module_scope]);
module_last_layer_name = ['concat_', module_scope];
lgraph = addLayers(lgraph, concatenationLayer(4, 4,'Name',module_last_layer_name));

lgraph = connectLayers(lgraph,['relu_1', module_scope], ['concat_', module_scope, '/in1']);
lgraph = connectLayers(lgraph,['relu_2', module_scope], ['concat_', module_scope, '/in2']);
lgraph = connectLayers(lgraph,['relu_3', module_scope], ['concat_', module_scope, '/in3']);
lgraph = connectLayers(lgraph,['relu_4', module_scope], ['concat_', module_scope, '/in4']);

end

function layers = non_linear_classifier(numClasses, scope)
% 3DmFV non linear classifier architecture
%Input:
%     numClasses: number of classes [int scalar]
%     scope: unique string 
% Output:
%       layers: nn layers vector 

drop_ratio = 0.3;
layers = [
    fc_bn_relu_dropout(1024, drop_ratio, ['1_', scope])
    fc_bn_relu_dropout(256, drop_ratio, ['2_', scope])
    fc_bn_relu_dropout(128, drop_ratio, ['3_', scope])
    fullyConnectedLayer(numClasses,'name', ['fc_4_', scope])
    softmaxLayer('name', ['softmax_', scope])
    classificationLayer('name', ['classifier_', scope])];
end

function layers = fc_bn_relu_dropout(n_neurons, drop_ratio, scope)
% Combine fully connected, batch normaliation, relu actuvation and drop out
%Input:
%   	 n_neurons: number of neurons in the fully connected layer [int scalar]
%       drop_ratio: ratio of outputs to drop between layers [double]
%    	scope: unique string 
% Output:
%       layers: nn layers vector 
layers=[
    fullyConnectedLayer(n_neurons,'name', ['fc_', scope])
    batchNormalizationLayer('name', ['bn_', scope])
    reluLayer('name', ['relu_', scope])
    dropoutLayer(drop_ratio,'name', ['dropout_', scope])
    ];
end