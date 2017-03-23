function [ net, info ] = run_experiment( varargin )

    % Default values for experiments
    opts.imdb = 'full_imdb';
    opts.arch = 'deepmask';
    opts.learningRate = 2e-6;
    opts.weightDecay = 0;
    opts.momentum = 0.9;
    opts.batchSize = 50;
    opts.numEpochs = 8;
    opts.saveInterval = 1;
    opts.continue = false;
    
    % Specific architecture variables must be initalized by the caller
    opts.net = struct();
    
    % Override with user-defined values
    opts = vl_argparse(opts,varargin);
    
    % Create experiment base name
    experimentName = [ 'lr' num2str(opts.learningRate), ...
                       '_wd' num2str(opts.weightDecay), ...
                       '_mom' num2str(opts.momentum), ...
                       '_batch' num2str(opts.batchSize)];
                                           
    % Add to experiment name the parameters specific to chosen architecture
    specificNames = fieldnames(opts.net);
    for i = 1 : numel(specificNames)
        experimentName = [experimentName '_' specificNames{i} num2str(opts.net.(specificNames{i}))];        
    end
    
    % Replace decimal separator with 'p', so that the resulting name is a valid path
    experimentName = strrep(experimentName,'.','p');
    
    % Create directory name
    expDirName = fullfile('data', 'experiments', opts.arch, opts.imdb, experimentName);
    
    % Run the experiment
    [net, info] = cnn_masknet('imdbPath',fullfile(pwd,'data','COCO_datasets',[opts.imdb '.mat']), ...
                    'expDir',expDirName, ...
                    'arch', opts.arch, ...
                    'net', opts.net, ...
                    'train',struct('learningRate', opts.learningRate, ...
                                   'numEpochs',  opts.numEpochs, ...
                                   'weightDecay', opts.weightDecay, ...
                                   'momentum', opts.momentum, ...
                                   'batchSize', opts.batchSize, ...
                                   'saveInterval', opts.saveInterval, ...
                                   'continue', opts.continue));    

end

