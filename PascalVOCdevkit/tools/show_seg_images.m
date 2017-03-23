addpath 'VOCcode';
VOCinit;

% Choose the part of the dataset to display
imgset = 'train';

% Read all image ids in the chosen set
ids = textread(sprintf(VOCopts.seg.imgsetpath,imgset),'%s');

% For each image
for i = 1 : numel(ids)
    % Get the paths to the images, segmentations and annotations
    imgpath = sprintf(VOCopts.imgpath,ids{i});
    annopath = sprintf(VOCopts.annopath,ids{i});
    clssegpath = sprintf(VOCopts.seg.clsimgpath,ids{i});
    objsegpath = sprintf(VOCopts.seg.instimgpath,ids{i});
    
    % Load the images, segmentations and annotations
    ann = PASreadrecord(annopath);
    img = imread(imgpath);
    clsseg = imread(clssegpath);
    objseg = imread(objsegpath);
    
    subplot(1,2,1);
    imshow(img);
    subplot(1,2,2);
    cMap = VOClabelcolormap(256);
    imshow(clsseg,cMap);
    
    waitforbuttonpress;
    
end