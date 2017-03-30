function [ net, batchFn ] = deepmask_bNorm_init( netOpts, trainOpts )

    % Default initalizations
    opts.net.maskSize = [224 224];
    opts.train.batchSize = 50;
    
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

    % Add the batch normalization layers
    net = insertBnorm(net, 1) ;
    net = insertBnorm(net, 6) ;
    net = insertBnorm(net, 11) ;
    net = insertBnorm(net, 14) ;
    net = insertBnorm(net, 17) ;
    net = insertBnorm(net, 20) ;
    net = insertBnorm(net, 24) ;
    
    % Tidy the net
    net = vl_simplenn_tidy(net);
    
    % Meta parameters
    net.meta.inputSize = [opts.net.maskSize(1) opts.net.maskSize(2) 3 opts.train.batchSize] ;
    
    % Return batch function
    batchFn = @(x,y)getBatchDeepmask(trainOpts,x,y);
end


function net = insertBnorm(net, l)

    assert(isfield(net.layers{l}, 'weights'));
    ndim = size(net.layers{l}.weights{1}, 4);
    layer = struct('type', 'bnorm', ...
                   'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
                   'learningRate', [1 1 0.05], ...
                   'weightDecay', [0 0]) ;
    net.layers{l}.biases = [] ;
    net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
    
end


