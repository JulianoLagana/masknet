function [ net ] = masknet3_init( varargin )

    % Default initalizations
    opts.batchSize = 50;
    opts.dropoutRate = 0.7;
    opts.maskSize = [224 224];
    opts.batchNormalization = 0;
    
    opts.mSize = 80;
    opts.nFeatures = 50;
    
    % Override default with user-specified values
    [opts,~] = vl_argparse(opts, varargin) ;
    
    M = opts.mSize;
    f = opts.nFeatures;
    
    % The first part is pre-initialized VGG network, with all the layers after the
    % 14th removed
    net = load_vgg_feature_computer('data/imagenet-vgg-m.mat');
    net.meta = [];
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);
    net.renameVar('x14','vgg_features');
    
    % Layers to concatenate VGG feature maps and partial mask
        
        % Constant grid generator for the upsampling layer
        grid = single(create_meshgrid([M, M], opts.batchSize));
        constantGridBlock = dagnn.Constant('value', grid);
        net.addLayer('constantGridGen',constantGridBlock,{},{'grid'});
    
        % Bilinear upsampling layer
        bilinearBlock1 = dagnn.BilinearSampler();
        net.addLayer('bilinear1', bilinearBlock1, {'pmask', 'grid'}, {'pmask_rsz'});
        
        % Reshape layer
        reshapeBlock1 = dagnn.Reshape('newDimensions',[1 1 M^2]);
        net.addLayer('reshape1',reshapeBlock1, {'pmask_rsz'},{'pmask_vec'},{});
    
    % New layers
    
        % Convolution layer
        convBlock6 = dagnn.Conv('size', [1 1 512 f], 'hasBias', true) ;
        net.addLayer('conv6', convBlock6, {'vgg_features'}, {'x14'}, {'conv6f', 'conv6b'}) ;

        % ReLU layer
        reluBlock6 = dagnn.ReLU() ;
        net.addLayer('relu6', reluBlock6, {'x14'}, {'x15'}, {}) ;
        
        % Reshape layer
        reshapeBlock2 = dagnn.Reshape('newDimensions',[1 1 13^2*f]);
        net.addLayer('reshape2',reshapeBlock2, {'x15'},{'x16'});
        
        % Concatenate feature maps with mask
        concatBlock = dagnn.Concat('dim',3);
        net.addLayer('concatenate', concatBlock, {'x16','pmask_vec'},{'x17'});
        
        % Huge fully connected layer
        convBlock7 = dagnn.Conv('size', [1 1 (13^2*f+M^2) 56^2], 'hasBias', true) ;
        net.addLayer('conv7', convBlock7, {'x17'}, {'x18'}, {'conv7f', 'conv7b'}) ;
        
        % Reshape layer
        reshapeBlock3 = dagnn.Reshape('newDimensions',[56 56 1]);
        net.addLayer('reshape3',reshapeBlock3, {'x18'},{'x19'});      

        % Constant grid generator for the upsampling layer
        grid2 = single(create_meshgrid(opts.maskSize, opts.batchSize));
        constantGridBlock2 = dagnn.Constant('value', grid2);
        net.addLayer('constantGridGen2',constantGridBlock2,{},{'grid2'});

        % Bilinear upsampling layer
        bilinearBlock2 = dagnn.BilinearSampler();
        net.addLayer('bilinear2', bilinearBlock2, {'x19', 'grid2'}, {'prediction'});

        % Logistic log-loss layer
        logLossBlock = dagnn.Loss('loss','logistic');
        net.addLayer('logloss', logLossBlock,{'prediction','gtMask'},{'objective'});

        % IoU error layer
        IoUblock = dagnn.Loss('loss','iouerror');
        net.addLayer('IoUerr', IoUblock, {'prediction','gtMask'},{'IoUerr'});

    % Randomly nitialize all parameters for the new layers
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
    
    % Meta parameters
    net.meta.inputSize = [opts.maskSize(1) opts.maskSize(2) 3 opts.batchSize] ;
    net.meta.trainOpts = [];

end