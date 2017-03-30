% SCRIPT TO TEST THE CORRECT FUNCTIONING OF THE run_fast_rcnn FUNCTION.

clear; clc; 

% Initialize PASCAL VOC devkit functions
VOCinit;

% Read all image ids in the training and validation Pascal VOC dataset
imgset = 'trainval';
ids = textread(sprintf(VOCopts.seg.imgsetpath,imgset),'%s');
    
% Use only the first 10 images
ids = ids(1:5);

for i = 1 : numel(ids)
    disp(i);
    % Get the paths to the images, segmentations and annotations
    imgpath = sprintf(VOCopts.imgpath,ids{i});
    annopath = sprintf(VOCopts.annopath,ids{i});
    clssegpath = sprintf(VOCopts.seg.clsimgpath,ids{i});
    objsegpath = sprintf(VOCopts.seg.instimgpath,ids{i});

    % Load the images, annotations and segmentations
    ann = PASreadrecord(annopath);
    img = imread(imgpath);
    clsseg = imread(clssegpath);
    objseg = imread(objsegpath);
    
    % Concatenate current image in the end of 4D matrix
    im{i} = img;
    
    %Create proposals for current image and save them in the cell matrix
    boxMatrix{i} = generateProposals(img);
end

% Run fast-rcnn in all images
disp('running fast-rcnn');
detections = run_fast_rcnn(im,boxMatrix);

% For each processed image
for i = 1 : numel(ids)
    
    % Change bbox conventions to MATLAB convention
    boxes = detections{i};
    for iBox = 1 : size(boxes,1)
        boxes(iBox,3) = boxes(iBox,3)-boxes(iBox,1);
        boxes(iBox,4) = boxes(iBox,4)-boxes(iBox,2);
    end
    boxes(:,1:4) = boxes(:,1:4)+1;

    % Show the image
    imshow(im{i});
    for k = 1 : size(boxes,1)
        boxes = double(boxes);
        rectangle('Position',boxes(k,1:4), 'EdgeColor','r');
        text(boxes(k,1), boxes(k,2)-5, num2str(k), 'FontSize', 10, 'Color', 'r');
    end
    
    waitforbuttonpress;
end