function [net, info] = cnn_masknet(varargin)
%CNN_MASKNET  Trains the masknet.

    % Initializations
    opts.expDir = fullfile(pwd, 'data', 'exp') ;
    opts.dataDir =  fullfile(pwd, 'data');
    opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
    opts.train = struct() ;
    opts.train.gpus = 1;

    % Default network parameters
    opts.arch = 'deepmask';
    opts.net.batchSize = 50 ;
    opts.net.dropoutRate = 0.4;

    % Default training parameters
    opts.train.numEpochs = 3 ;
    opts.train.learningRate = 0.00001/5 ;
    opts.train.weightDecay = 0.00005 ;
    opts.train.momentum = 0.9 ;
    opts.train.continue = false;

    % Override any options with the user-defined values
    opts = vl_argparse(opts, varargin) ;

    % Reset the GPU
    fprintf('%s: resetting GPU\n', mfilename) ;
    clear mex ;
    clear vl_tmove vl_imreadjpeg ;
    disp(gpuDevice(opts.train.gpus)) ;

    % Load the chosen network architecture
    isDag = false;
    addpath archs;
    switch opts.arch
        case 'deepmask'
            net = deepmask_init(opts.net);
        case 'deepmask_dropoutBefore'
            net = deepmask_dropoutBefore_init(opts.net);
        case 'deepmask_dropoutAfter'
            net = deepmask_dropoutAfter_init(opts.net);
        case 'deepmask_bNorm'
            net = deepmask_bNorm_init(opts.net);
        case 'masknet'
            net = masknet_init(opts.net);
            isDag = true;
    end

    % Load the imdb file
    if exist(opts.imdbPath, 'file')
      imdb = matfile(opts.imdbPath) ;
    else
      error('imdb.mat file was not found')
    end

    % Separate data in training and validation samples (because of the way a
    % matfile object accesses data, they cannot be randomly sampled)
    temp = whos(imdb);
    nImages = temp(1).size(4);
    training_ratio = 0.7;
    validation_ratio = 0.15;
    set = zeros(nImages,1);
    for i = 1 : nImages

        % Training set
        if i <= nImages*training_ratio
            set(i) = 1;

        % Validation set
        elseif i <= nImages*(training_ratio + validation_ratio)
            set(i) = 2;

        % Test set
        else
            set(i) = 3;
        end

    end

    % Choose batch function
    if strcmp(opts.arch,'masknet')
        batchFn = @(x,y) getBatchMasknet(opts.train,x,y);
    else
        batchFn = @(x,y)getBatch(x,y);
    end

    % Train
    if isDag
        [net, info] = cnn_train_dag(net, imdb, batchFn, ...
          'expDir', opts.expDir, ...
          opts.train, ...
          'batchSize', opts.net.batchSize, ...
          'train', find(set == 1), ...
          'val', find(set == 2)) ;
    else
        [net, info] = cnn_train(net, imdb, batchFn, ...
          'expDir', opts.expDir, ...
          opts.train, ...
          'batchSize', opts.net.batchSize, ...
          'train', find(set == 1), ...
          'val', find(set == 2)) ;    
    end

end


% --------------------------------------------------------------------
function [images, masks] = getBatch(imdb, batch)
% --------------------------------------------------------------------
    images = single(imdb.imdb(:,:,:,batch));
    masks = single(imdb.masks(:,:,1,batch));
end

% --------------------------------------------------------------------
function inputs = getBatchMasknet(opts,imdb, batch)
% --------------------------------------------------------------------
    images = single(imdb.imdb(:,:,:,batch));
    
    partial_masks = single(imdb.partial_masks(:,:,1,batch));
    partial_masks(partial_masks == 0) = -1;
    
    masks = single(imdb.masks(:,:,1,batch));
    masks(masks == 0) = -1;
    
    if numel(opts.gpus) > 0
        images = gpuArray(images);
        partial_masks = gpuArray(partial_masks);    
        masks = gpuArray(masks);  
    end
    
    inputs = {'input',images,'gtMask',masks};
end