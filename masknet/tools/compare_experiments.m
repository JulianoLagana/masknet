function [] = compare_experiments( experimentDirs, varargin )

    nExperiments = numel(experimentDirs);
    trainingData = cell(1,nExperiments);
    validationData = cell(1,nExperiments);
    
    opts.error = 'IoUerr';
    opts = vl_argparse(opts,varargin);
    
    % Preppend 'data\experiments\' to all directories
    for i = 1 : nExperiments
        experimentDirs{i} = ['data\experiments\' experimentDirs{i}];
    end
    
    % For each experiment directory, save the training and validation data
    for i = 1 : nExperiments
        experimentDir = experimentDirs{i};
        
        % Find the last save
        last = findLastCheckpoint(experimentDir);
        
        % Save training and validation data
        load([experimentDir '\net-epoch-' num2str(last) '.mat'], 'stats');
        trainingData{i} = [stats.train.(opts.error)];
        validationData{i} = [stats.val.(opts.error)];

    end
    
    expNames = strrep(experimentDirs,'data\experiments\','');
    expNames = strrep(expNames,'_','-');
    colors = distinguishable_colors(nExperiments,[1 1 1]);
    
    % Plot all experiments
    figure; 
    hold on;
    for i = 1 : nExperiments
        if strcmp(opts.error,'IoUerr')
            plot(1-trainingData{i},'-.','Color',colors(i,:));
        else
            plot(trainingData{i},'-.','Color',colors(i,:));
        end
    end
    legend(expNames);
    for i = 1 : nExperiments
        if strcmp(opts.error,'IoUerr')
            plot(1-validationData{i},'Color',colors(i,:));
        else
            plot(validationData{i},'Color',colors(i,:));
        end            
    end
    grid;
    
end


% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
    list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
    tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
    epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
    epoch = max([epoch 0]) ;
end