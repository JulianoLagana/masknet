experimentDirs = ...
    {'masknet3\VOC2012\pascal_imdb\lr2e-06_wd0_mom0p9_batch30_M224_f300', ...
    'deepmask_dag\VOC2012\pascal_imdb\lr1e-06_wd5e-05_mom0p9_batch40', ...
    'masknet3\VOC2012\pascal_imdb\lr1e-06_wd0_mom0p9_batch30_preInitModelPathdata!experiments!masknet3!COCO_datasets!centered_imdb!lr2e-06_wd0_mom0p9_batch30_M224_f300!net-epoch-5pmat', ...
    'deepmask_dag\VOC2012\pascal_imdb\lr1e-06_wd5e-05_mom0p9_batch40_preInitModelPathdata!experiments!deepmask_dag!COCO_datasets!centered_imdb!lr1e-06_wd5e-05_mom0p9_batch40!net-epoch-3pmat'};   

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

expNames = {'Merging strategy 5', 'No partial masks','Merging strategy 5 (P)', 'No partial masks (P)'};
colors = distinguishable_colors(nExperiments,[1 1 1]);

% Plot all experiments
figure; 
hold on;
for i = 1 : nExperiments
    plot(validationData{i},'-','Color',colors(i,:));
end
legend(expNames);
for i = 1 : nExperiments
    plot(trainingData{i},'--','Color',colors(i,:));
end

ylabel 'IoU error'
xlabel 'No. of epochs'
grid;


% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
    list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
    tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
    epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
    epoch = max([epoch 0]) ;
end