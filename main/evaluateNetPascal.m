clear, clc;

% Initialize PASCAL VOC devkit functions
VOCinit;
cat_names = {'background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat', ...
'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant', ...
'sheep','sofa','train','tvmonitor'};

% Parameters
opts.batchSize = 100;
opts.confidenceLevels = linspace(0,1,100);
opts.masknetPath = 'data\experiments\masknet3\VOC2012\pascal_imdb\lr1e-06_wd0_mom0p9_batch30_preInitModelPathdata!experiments!masknet3!COCO_datasets!centered_imdb!lr2e-06_wd0_mom0p9_batch30_M224_f300!net-epoch-5pmat/net-epoch-4.mat';

% Initialize aggregators
truePositives = zeros(numel(opts.confidenceLevels),numel(cat_names)-1);
falsePositives = zeros(numel(opts.confidenceLevels),numel(cat_names)-1);
falseNegatives = zeros(numel(opts.confidenceLevels),numel(cat_names)-1);

% Other initializations
batchNumber = 1;

% Choose all validation images
ids = textread(sprintf(VOCopts.seg.imgsetpath,'val'),'%s');
nImages = numel(ids);

while ~isempty(ids)
    
    batchSize = min(opts.batchSize,numel(ids));
    
    idsToProcess = ids(1:batchSize);
    ids(1:batchSize) = [];
    
    fprintf('------------- Batch %d/%d -------------\n', batchNumber, ceil(nImages/opts.batchSize));

    % Load the images in a cell array
    imgs = cell(1,batchSize);
    for i = 1 : batchSize
        imgpath = sprintf(VOCopts.imgpath,idsToProcess{i});
        imgs{i} = imread(imgpath);
    end

    % Run full net
    instances = run_full_net(imgs, idsToProcess, 'verbose', true, 'masknetPath', opts.masknetPath);
    
    % Aggregate values for computing MAP score
    tic;
    fprintf('aggregating scores...')
    [truePositives,falsePositives,falseNegatives] = aggregatePRPascal(instances,truePositives,falsePositives,falseNegatives, 'confidenceLevels',opts.confidenceLevels);    
    fprintf(' (%.3fs)\n',toc);
    
    batchNumber = batchNumber + 1;
    fprintf('\n');
    
end

save aggregators truePositives falsePositives falseNegatives;