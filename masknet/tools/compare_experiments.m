function [] = compare_experiments( experimentDirs )

    nExperiments = numel(experimentDirs);
    trainingData = cell(1,nExperiments);
    validationData = cell(1,nExperiments);
    
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
        trainingData{i} = [stats.train.IoUerr];
        validationData{i} = [stats.val.IoUerr];

    end
    
    expNames = strrep(experimentDirs,'data\experiments\','');
    expNames = strrep(expNames,'_','-');
    colors = distinguishable_colors(nExperiments,[1 1 1]);
    
    % Plot all experiments
    figure; 
    hold on;
    for i = 1 : nExperiments
        plot(trainingData{i},'-.','Color',colors(i,:));
    end
    legend(expNames);
    for i = 1 : nExperiments
        plot(validationData{i},'Color',colors(i,:));
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