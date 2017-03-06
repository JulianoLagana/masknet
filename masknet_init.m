function [ net ] = masknet_init( batchSize )

    % The first part is just a VGG network, with all the layers after the
    % 14th removed
    net = load_vgg_feature_computer('data/imagenet-vgg-m.mat');
    net.meta = [];
    
    f = 1/100;
    % Conv + ReLU layers
    net.layers{end+1} = struct('type', 'conv',...
                               'weights', {{f*randn(1,1,512,512, 'single'), zeros(1, 512, 'single')}} , ...
                               'stride', 1, ...
                               'pad', 0) ;
    net.layers{end+1} = struct('type','relu');
    
    % Conv layer
    net.layers{end+1} = struct('type','conv',...
                               'weights', {{f*randn(13,13,512,512,'single'), zeros(1,512,'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;
                           
    % Conv layer                           
    net.layers{end+1} = struct('type','conv',...
                               'weights', {{f*randn(1,1,512,56^2,'single'), zeros(1,56^2,'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;  
                           
    % Reshape layer                           
    net.layers{end+1} = struct('type','reshape',...
                               'newDim', [56,56,1]);
                           
    % Bilinear upsampling layer
    outDim = [224,224];
    grid = single(create_meshgrid(outDim, batchSize));
    grid = gpuArray(grid);
    net.layers{end+1} = struct('type','bilinear',...
                               'grid',grid);     
           
    % Logistic log-loss layer                           
    net.layers{end+1} = struct('type','loss','loss','logistic','class', []);                          
                        
    % Tidy the net
    net = vl_simplenn_tidy(net);
    
    % Meta parameters
    net.meta.inputSize = [224 224 3 batchSize] ;
    net.meta.trainOpts = [];

end

