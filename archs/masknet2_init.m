function [ net ] = masknet2_init( netOpts, trainOpts )

    % Default initalizations
    opts.train.batchSize = 50;
    opts.net.maskSize = [224 224];
    
    % Override default with user-specified values
    opts.net = vl_argparse(opts.net, netOpts) ;
    [opts.train, ~] = vl_argparse(opts.train, trainOpts) ;
    
    % The first part is pre-initialized VGG network, with all the layers after the
    % 14th removed
    net = load_vgg_feature_computer('data/imagenet-vgg-m.mat');
    net.meta = [];
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);
    net.renameVar('x14','vgg_features');
    
    % Layers to concatenate VGG feature maps and partial mask
        
        % Constant grid generator for the upsampling layer
        grid = single(create_meshgrid([13 13], opts.train.batchSize));
        constantGridBlock = dagnn.Constant('value', grid);
        net.addLayer('constantGridGen',constantGridBlock,{},{'grid'});
    
        % Bilinear upsampling layer
        bilinearBlock = dagnn.BilinearSampler();
        net.addLayer('bilinear', bilinearBlock, {'pmask', 'grid'}, {'pmask_rsz'});
        
        % Concatenate feature maps with mask
        concatBlock = dagnn.Concat('dim',3);
        net.addLayer('concatenate', concatBlock, {'vgg_features','pmask_rsz'},{'x14_before'});
        
        % Convolute them
        convBlockMask = dagnn.Conv('size',[1 1 513 512], 'hasBias', true);
        net.addLayer('convMask', convBlockMask, {'x14_before'}, {'x14'},{'convMaskf', 'convMaskb'});
    
    % New layers
    
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
        grid2 = single(create_meshgrid(opts.net.maskSize, opts.train.batchSize));
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
    l1 = net.getLayerIndex('convMask');
    l2 = net.getLayerIndex('IoUerr');
    net.initParams(l1:l2);
    
    f = 1/100;
    
    iConvMaskf = net.getParamIndex('convMaskf');
    sz = size(net.params(iConvMaskf).value);
    net.params(iConvMaskf).value = f*randn(sz,'single');
    
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
    net.meta.inputSize = [opts.net.maskSize(1) opts.net.maskSize(2) 3 opts.train.batchSize] ;

end
