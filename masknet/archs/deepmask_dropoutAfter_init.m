function [ net, batchFn ] = deepmask_dropoutAfter_init( netOpts, trainOpts )

    % Default initalizations
    opts.train.batchSize = 50;
    opts.net.dropoutRate = 0.7;
    opts.net.maskSize = [224 224];
    
    % Override default with user-specified values
    opts.net = vl_argparse(opts.net, netOpts) ;
    [opts.train, ~] = vl_argparse(opts.train, trainOpts) ;

    % The first part is just a VGG network, with all the layers after the
    % 14th removed
    net = load_vgg_feature_computer('data/models/imagenet-vgg-m.mat');
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
                           
    % Dropout layer
    net.layers{end+1} = struct('type', 'dropout', ...
                               'rate', opts.net.dropoutRate);                            
                           
    % Conv layer                           
    net.layers{end+1} = struct('type','conv',...
                               'weights', {{f*randn(1,1,512,56^2,'single'), zeros(1,56^2,'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;  
                           
    % Reshape layer                           
    net.layers{end+1} = struct('type','reshape',...
                               'newDim', [56,56,1]);
                           
    % Bilinear upsampling layer
    grid = single(create_meshgrid(opts.net.maskSize, opts.train.batchSize));
    grid = gpuArray(grid);
    net.layers{end+1} = struct('type','bilinear',...
                               'grid',grid);     
           
    % Logistic log-loss layer                           
    net.layers{end+1} = struct('type','loss','loss','logistic','class', []);                          
    
    % Tidy the net
    net = vl_simplenn_tidy(net);
    
    % Meta parameters
    net.meta.inputSize = [opts.net.maskSize(1) opts.net.maskSize(2) 3 opts.train.batchSize] ;
    net.meta.trainOpts = [];

    % Return batch function
    batchFn = @(x,y)getBatchDeepmask(trainOpts,x,y);
end

% --------------------------------------------------------------------
function [images, masks] = getBatchDeepmask(opts, imdb, batch)
% --------------------------------------------------------------------
    images = single(imdb.imdb(:,:,:,batch));
    masks = single(imdb.masks(:,:,1,batch));
    masks(masks == 0) = -1;    
    
    if numel(opts.gpus) > 0
        images = gpuArray(images);
        masks = gpuArray(masks);
    end
end