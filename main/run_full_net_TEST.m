% Initialize PASCAL VOC devkit functions
VOCinit;

% Read first n image ids in the training and validation Pascal VOC dataset
n = 100;
ids = textread(sprintf(VOCopts.seg.imgsetpath,'trainval'),'%s');
ids = ids(1:n);

% Load the images in a cell array
for i = 1 : n
    imgpath = sprintf(VOCopts.imgpath,ids{i});
    imgs{i} = imread(imgpath);
end

% Run full net
run_full_net(imgs, 'verbose', true);