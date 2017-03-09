function [ net ] = masknet_init( varargin )

    % Default initalizations
    opts.batchSize = 50;
    opts.dropoutRate = 0.7;
    opts.maskSize = [224 224];
    opts.batchNormalization = 0;
    
    % Override default with user-specified values
    opts = vl_argparse(opts, varargin) ;

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
    
    % Dropout layer
    net.layers{end+1} = struct('type', 'dropout', ...
                               'rate', opts.dropoutRate);  
    
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
    grid = single(create_meshgrid(opts.maskSize, opts.batchSize));
    grid = gpuArray(grid);
    net.layers{end+1} = struct('type','bilinear',...
                               'grid',grid);     
           
    % Logistic log-loss layer                           
    net.layers{end+1} = struct('type','loss','loss','logistic','class', []);                          

    % If requested, add the batch normalization layers
    if opts.batchNormalization
        net = insertBnorm(net, 1) ;
        net = insertBnorm(net, 6) ;
        net = insertBnorm(net, 11) ;
        net = insertBnorm(net, 14) ;
        net = insertBnorm(net, 17) ;
        net = insertBnorm(net, 20) ;
        net = insertBnorm(net, 25) ;
    end
    
    % Tidy the net
    net = vl_simplenn_tidy(net);
    
    % Meta parameters
    net.meta.inputSize = [opts.maskSize(1) opts.maskSize(2) 3 opts.batchSize] ;
    net.meta.trainOpts = [];

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
