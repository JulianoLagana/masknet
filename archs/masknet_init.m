function [ net ] = masknet_init( varargin )

    % Default initalizations
    opts.batchSize = 50;
    opts.dropoutRate = 0.7;
    opts.maskSize = [224 224];
    opts.batchNormalization = 0;
    
    % Override default with user-specified values
    opts = vl_argparse(opts, varargin) ;

    
    % The first part is pre-initialized VGG network, with all the layers after the
    % 14th removed
    net = load_vgg_feature_computer('data/imagenet-vgg-m.mat');
    net.meta = [];
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);
    
    % Convolution layer
    convBlock6 = dagnn.Conv('size', [1 1 512 512], 'hasBias', true) ;
    net.addLayer('conv6', convBlock6, {'x14'}, {'x15'}, {'conv6f', 'conv6b'}) ;
    
    % ReLU layer
    reluBlock6 = dagnn.ReLU() ;
    net.addLayer('relu6', reluBlock6, {'x15'}, {'x16'}, {}) ;

    % Convolution layer
    convBlock7 = dagnn.Conv('size', [13 13 512 512], 'hasBias', true) ;
    net.addLayer('conv7', convBlock7, {'x16'}, {'x17'}, {'conv7f', 'conv7b'}) ;
    
    % Convolution layer
    convBlock8 = dagnn.Conv('size', [1 1 512 56^2], 'hasBias', true) ;
    net.addLayer('conv8', convBlock8, {'x17'}, {'x18'}, {'conv8f', 'conv8b'}) ;
    
    % Reshape layer
    reshapeBlock = dagnn.Reshape('newDimensions',[56 56 1]);
    net.addLayer('reshape',reshapeBlock, {'x18'},{'x19'},{});
    
    % Constant grid generator for the upsampling layer
    grid = single(create_meshgrid(opts.maskSize, opts.batchSize));
    constantGridBlock = dagnn.Constant('value', grid);
    net.addLayer('constantGridGen',constantGridBlock,{},{'grid'});
    
    % Bilinear upsampling layer
    bilinearBlock = dagnn.BilinearSampler();
    net.addLayer('bilinear', bilinearBlock, {'x19', 'grid'}, {'prediction'});
    
    % Logistic log-loss layer
    logLossBlock = dagnn.Loss('loss','logistic');
    net.addLayer('logloss', logLossBlock,{'prediction','gtMask'},{'objective'});
    
    % IoU error layer
    IoUblock = dagnn.Loss('loss','iouerror');
    net.addLayer('IoUerr', IoUblock, {'prediction','gtMask'},{'IoUerr'});
    
    % Initialize all parameters (except for the VGG ones)
    l1 = net.getLayerIndex('conv6');
    l2 = net.getLayerIndex('IoUerr');
    net.initParams(l1:l2);
    
    f = 1/100;
    iConv6f = net.getParamIndex('conv6f');
    sz = size(net.params(iConv6f).value);
    net.params(iConv6f).value = f*randn(sz,'single');
    
    iConv7f = net.getParamIndex('conv7f');
    sz = size(net.params(iConv7f).value);
    net.params(iConv7f).value = f*randn(sz,'single');
    
    iConv8f = net.getParamIndex('conv8f');
    sz = size(net.params(iConv8f).value);
    net.params(iConv8f).value = f*randn(sz,'single');
    
    
    
    
    
    % Meta parameters
    net.meta.inputSize = [opts.maskSize(1) opts.maskSize(2) 3 opts.batchSize] ;
    net.meta.trainOpts = [];

end
