function [ net, info ] = run_experiment( varargin )

    % Default values for experiments
    opts.imdb = 'full_imdb';
    opts.arch = 'deepmask';
    opts.learningRate = 2e-6;
    opts.weightDecay = 0;
    opts.batchSize = 50;
    opts.numEpochs = 8;
    opts.continue = false;
    opts.dropoutRate = 0; % no dropout
    
    % Override with user-defined values
    opts = vl_argparse(opts,varargin);
    
    % Create experiment base name
    experimentName = [strrep(opts.imdb,'_','') '_lr' num2str(opts.learningRate), ...
                                               '_wd' num2str(opts.weightDecay), ...
                                               '_batch' num2str(opts.batchSize)];
                                           
    % Add to experiment name parameters specific to chosen architecture                                           
    switch opts.arch
        case 'deepmask_dropoutAfter'
            experimentName = [experimentName '_drop' num2str(opts.dropoutRate)];
        case 'deepmask_dropoutBefore'
            experimentName = [experimentName '_drop' num2str(opts.dropoutRate)];
    end
    
    % Replace commas with 'p', so that the resulting name is a valid path
    experimentName = strrep(experimentName,'.','p');
    
    % Run the experiment
    [net, info] = cnn_masknet('imdbPath',fullfile(pwd,'data',[opts.imdb '.mat']), ...
                    'expDir',fullfile(pwd, 'data', 'experiments', opts.arch, experimentName), ...
                    'arch', opts.arch, ...
                    'net', struct('batchSize', opts.batchSize, ...
                                  'dropoutRate', opts.dropoutRate), ...
                    'train',struct('learningRate', opts.learningRate, ...
                                   'numEpochs',  opts.numEpochs, ...
                                   'weightDecay', opts.weightDecay, ...
                                   'continue', opts.continue));    

end

