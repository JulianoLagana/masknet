function [net, info] = cnn_masknet(varargin)
%CNN_MASKNET  Trains the masknet.

% Initializations
opts.network = [] ;
opts.expDir = fullfile(pwd, 'data', 'exp') ;
opts.dataDir =  fullfile(pwd, 'data');
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
opts.train = struct() ;
opts.train.gpus = 1;

% Training parameters
opts.train.numEpochs = 3 ;
opts.train.learningRate = 0.00001 ;
opts.train.weightDecay = 0.005 ;
opts.train.momentum = 0.9 ;
opts.train.batchSize = 200 ;

% Override any options with the user-defined values
opts = vl_argparse(opts, varargin) ;

% Reset the GPU
fprintf('%s: resetting GPU\n', mfilename) ;
clear mex ;
clear vl_tmove vl_imreadjpeg ;
disp(gpuDevice(opts.train.gpus)) ;

% Load the network (or read from arguments)
if isempty(opts.network)
  net = masknet_init(opts.train.batchSize) ;
else
  net = opts.network ;
  opts.network = [] ;
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

% Train
[net, info] = cnn_train(net, imdb, @(x,y)getBatch(x,y), ...
  'expDir', opts.expDir, ...
  opts.train, ...
  'train', find(set == 1), ...
  'val', find(set == 2)) ;

end


% --------------------------------------------------------------------
function [images, masks] = getBatch(imdb, batch)
% --------------------------------------------------------------------
    images = single(imdb.imdb(:,:,:,batch));
    masks = single(imdb.masks(:,:,1,batch));
end