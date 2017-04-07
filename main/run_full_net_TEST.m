% Initialize PASCAL VOC devkit functions
VOCinit;

% Read first n image ids in the training and validation Pascal VOC dataset
n = 10;
ids = textread(sprintf(VOCopts.seg.imgsetpath,'trainval'),'%s');
ids = ids(1:n);

% Load the images in a cell array
for i = 1 : n
    imgpath = sprintf(VOCopts.imgpath,ids{i});
    imgs{i} = imread(imgpath);
end

% Run full net
masknetPath = 'data\experiments\masknet3\VOC2012\pascal_imdb\lr1e-06_wd0_mom0p9_batch30_preInitModelPathdata!experiments!masknet3!COCO_datasets!centered_imdb!lr2e-06_wd0_mom0p9_batch30_M224_f300!net-epoch-5pmat/net-epoch-4.mat';
run_full_net(imgs, ids, 'verbose', true, 'masknetPath', masknetPath, 'debug', true);