function [ net, info ] = run_experiment( varargin )

    % Default values for experiments
    opts.imdb = 'full_imdb';
    opts.modelName = [];
    opts.learningRate = 2e-6;
    opts.weightDecay = 0;
    opts.batchSize = 50;
    opts.numEpochs = 8;
    opts.continue = false;
    opts.dropoutRate = 1; % no dropout
    opts.batchNormalization = 0; % no batch normalization
    
    % Override with user-defined values
    opts = vl_argparse(opts,varargin);
    
    % Add '+bNorm' to model name if batch normalization is being used
    if opts.batchNormalization
        opts.modelName = [opts.modelName '+bNorm'];
    end
    
    % Create experiment name
    experimentName = [strrep(opts.imdb,'_','') '_' opts.modelName, ...
                                               '_lr' num2str(opts.learningRate), ...
                                               '_wd' num2str(opts.weightDecay), ...
                                               '_drop' num2str(opts.dropoutRate), ...
                                               '_batch' num2str(opts.batchSize)];
    experimentName = strrep(experimentName,'.','p');
    
    % Run the experiment
    [net, info] = cnn_masknet('imdbPath',fullfile(pwd,'data',[opts.imdb '.mat']), ...
                    'expDir',fullfile(pwd, 'data', 'experiments', experimentName), ...
                    'net', struct('batchSize', opts.batchSize, ...
                                  'dropoutRate', opts.dropoutRate, ...
                                  'batchNormalization', opts.batchNormalization), ...
                    'train',struct('learningRate', opts.learningRate, ...
                                   'numEpochs',  opts.numEpochs, ...
                                   'weightDecay', opts.weightDecay, ...
                                   'continue', opts.continue));    

end

