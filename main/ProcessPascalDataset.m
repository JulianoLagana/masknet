clear; clc; rng(1);
profile on;

% Parameters
opts.savePath = 'data/VOC2012/SegmentationFCN';
opts.saveSize = [224 224];
DEBUG = false;

% Initialize PASCAL VOC devkit functions
VOCinit;

% Read all image ids in the training and validation Pascal VOC dataset
imgset = 'trainval';
allIds = textread(sprintf(VOCopts.seg.imgsetpath,imgset),'%s');

% Shuffle the image ids
allIds = allIds(randperm(numel(allIds)));

% DEBUG
%allIds = allIds(1:100);

% Create .mat file and matfile object to write dataset
delete 'pascal_imdb.mat';
file = matfile('pascal_imdb.mat');

% Create buffer
bufferSize = 100; % smallest possible value is 2, because of the way matfile initializes variables
shuffleIdx = randperm(bufferSize);
w = opts.saveSize(1);
h = opts.saveSize(2);
buffer1(1:w , 1:h , 3, bufferSize) = uint8(0);
buffer2(1:w , 1:h , 1, bufferSize) = single(0);
buffer3(1:w , 1:h , 1, bufferSize) = single(0);
n = 0;
nImagesSaved = 0;

nImages = numel(allIds);
disp([num2str(nImages) ' images in the ' imgset ' set.']);

% Process batches of images each iteration
batchSize = 100;
currentBatchNumber = 1;
nBatchesInTotal = ceil(nImages/batchSize);

% Initialize progress bar
progressTick = max(1,round(nBatchesInTotal/100));
handleWaitBar = waitbar(0,'Please wait.');

% While there are still remaining images
while ~isempty(allIds)
    
    tic;    
    disp(['--------- Batch ' num2str(currentBatchNumber) '/' num2str(nBatchesInTotal) ' ---------']);
    
    % Grab a batch of images
    m = min(batchSize, numel(allIds));
    ids = allIds(1:m);
    
    % REMOVE THESE IDS
    allIds(1:m) = [];
    
    % Initializations
    anns = cell(1,m);
    imgs = cell(1,m);
    clssegs = cell(1,m);
    objsegs = cell(1,m);
    gtBoxes = cell(1,m);
    props  = cell(1,m);    
    
    disp('Loading images and calculating initial proposals');
    % For each image in the batch
    for i = 1 : numel(ids)
        
        % Get the path to the image, annotations and segmentation
        imgpath = sprintf(VOCopts.imgpath,ids{i});
        annopath = sprintf(VOCopts.annopath,ids{i});
        clssegpath = sprintf(VOCopts.seg.clsimgpath,ids{i});
        objsegpath = sprintf(VOCopts.seg.instimgpath,ids{i});

        % Load the image, annotation and segmentation
        anns{i} = PASreadrecord(annopath);
        imgs{i} = imread(imgpath);
        clssegs{i} = imread(clssegpath);
        objsegs{i} = imread(objsegpath);
    
        % Get ground truth bounding boxes and put them in MATLAB's convention 
        % (i.e. [x1,y1,x2,y2]). Note: PASCAL bboxes are 1-based.
        gtBoxes{i} = reshape([anns{i}.objects.bbox],4,[])';
        for iBox = 1 : size(gtBoxes,1)
            gtBoxes{i}(iBox,3) = gtBoxes{i}(iBox,3)-gtBoxes{i}(iBox,1);
            gtBoxes{i}(iBox,4) = gtBoxes{i}(iBox,4)-gtBoxes{i}(iBox,2);
        end
        
        % Generate proposals for bounding boxes (to be used by fast-rcnn)
        props{i} = generateProposals(imgs{i}); % this outputs 0-based bboxes
        
    end
    % Segment the images using FCN-8s
    disp('segmenting...');
    segFCN = run_fcn_8s(imgs, 'gpu', 1);
    
    % Find bboxes using Fast-rcnn. Note: Fast-rcnn boxes are 0-based.
    disp('rcnn...');
    boxes = run_fast_rcnn(imgs,props);
    
    % Put predicted bboxes in MATLAB's convention.
    for iImage = 1 : numel(boxes)
       for iBox = 1 : size(boxes{iImage},1)
          boxes{iImage}(iBox,3) = boxes{iImage}(iBox,3)-boxes{iImage}(iBox,1);
          boxes{iImage}(iBox,4) = boxes{iImage}(iBox,4)-boxes{iImage}(iBox,2);
       end
       boxes{iImage}(:,1:4) = boxes{iImage}(:,1:4)+1;   
    end

    disp('Cropping and generating examples...');
    
    % For each image in the batch
    for i = 1 : numel(ids)
        
        % For each bbox found for that image
        for j = 1 : size(boxes{i},1)

            % Generate the image patch corresponding to the current bbox
            patch = cutPatch(imgs{i},boxes{i}(j,1:4));
            patch = imresize(patch, opts.saveSize);

            % Generate the ground truth corresponding to the current bbox
            gtMask = generateGtMask(boxes{i}(j,:), gtBoxes{i}, anns{i}, objsegs{i}, opts.saveSize);        

            % Generate the partial mask corresponding to the current bbox
            pMask = generatePartialMask(boxes{i}, j, segFCN{i}, opts.saveSize); 

            % DEBUG
            if(DEBUG)
                plotDebugInfo(imgs{i},clssegs{i},segFCN{i},boxes{i},gtBoxes{i},patch,pMask,gtMask,j);
            end

            % Put the masks in the format expected by matconvnet
            gtMask(gtMask == 0) = -1;
            gtMask(gtMask == 2) = 0;
            pMask(pMask == 0) = -1;

            % Save them in the buffers
            n = n + 1;
            buffer1(:,:,:,shuffleIdx(n)) = patch;
            buffer2(:,:,1,shuffleIdx(n)) = pMask;
            buffer3(:,:,1,shuffleIdx(n)) = gtMask;

            % If buffer is full, save to file and "empty" it
            if n == bufferSize

                % Debug
                disp('writing to file');
                % Determine if this is the first save
                varlist = whos(file);
                if numel(varlist) < 3
                    % If it is, we must create the variables without using the
                    % colon operator
                    file.imdb = buffer1;
                    file.partial_masks = buffer2;
                    file.masks = buffer3;
                    n = 0;
                    nImagesSaved = nImagesSaved + bufferSize;
                    shuffleIdx = randperm(bufferSize);
                else
                    % If not, determine how many images were already saved, and
                    % start saving from there
                    file.imdb(: , : , : , nImagesSaved+1 : nImagesSaved+n) = buffer1;
                    file.partial_masks(:,:, 1, nImagesSaved+1 : nImagesSaved+n) = buffer2;
                    file.masks(:,:, 1, nImagesSaved+1 : nImagesSaved+n) = buffer3;
                    n = 0;
                    nImagesSaved = nImagesSaved + bufferSize;
                    shuffleIdx = randperm(bufferSize);
                end

            end
        end
    end
    
    disp('done.');
    disp(['time spent: ' num2str(toc) 's']);
    
    % If it's the right time, update the progress bar
    if mod(currentBatchNumber,progressTick) == 0
        progress = currentBatchNumber/nBatchesInTotal;
        msg = sprintf('Please wait: %i%% complete',round(progress*100));
        waitbar(progress,handleWaitBar, msg);
    end
    
    % Update batch number
    currentBatchNumber = currentBatchNumber + 1;
end

% Empty the buffer by saving the remaining contents to file
if n > 0
    
    % Remove elements that are not from the current save
    idxsSaved = sort(shuffleIdx(1:n));
    buffer1 = buffer1(:,:,:,idxsSaved);
    buffer2 = buffer2(:,:,1,idxsSaved);
    buffer3 = buffer3(:,:,1,idxsSaved);
    
    % Determine if this is the first save
    varlist = whos(file);
    if numel(varlist) < 3
        % If it is, we must create the variables without using the
        % colon operator
        file.imdb = buffer1;
        file.partial_masks = buffer2;
        file.masks = buffer3;
        nImagesSaved = nImagesSaved + n;
        n = 0;
    else
        % If not, determine how many images were already saved, and
        % start saving from there
        file.imdb(: , : , : , nImagesSaved+1 : nImagesSaved+n) = buffer1(:,:,:,1:n);
        file.partial_masks(:,:, 1, nImagesSaved+1 : nImagesSaved+n) = buffer2(:,:,1,1:n);
        file.masks(:,:, 1, nImagesSaved+1 : nImagesSaved+n) = buffer3(:,:,1,1:n);
        nImagesSaved = nImagesSaved + n;
        n = 0;
    end
    
end
close(handleWaitBar);
profile viewer;


function plotDebugInfo(img, clsseg, segFCN, boxes, gtBoxes, patch, pMask, gtMask, j)

        % DEBUGGG
        cMap = VOClabelcolormap(256);
        subplot(2,3,1)
        imshow(img);
        for k = 1 : size(boxes,1)
            boxes = double(boxes);
            rectangle('Position',boxes(k,1:4), 'EdgeColor','r');
            text(boxes(k,1), boxes(k,2)-5, num2str(k), 'FontSize', 10, 'Color', 'r');
        end
        for k = 1 : size(gtBoxes,1)
            gtBoxes = double(gtBoxes);
            rectangle('Position',gtBoxes(k,1:4), 'EdgeColor', 'b');
            text(gtBoxes(k,1), gtBoxes(k,2)-5, num2str(k), 'FontSize', 10, 'Color', 'b');
        end
        title originalImage;
        
        subplot(2,3,2);
        imshow(clsseg,cMap);
        for k = 1 : size(boxes,1)
            boxes = double(boxes);
            rectangle('Position',boxes(k,1:4), 'EdgeColor','r');
            text(boxes(k,1), boxes(k,2)-5, num2str(k), 'FontSize', 10, 'Color', 'r');
        end
        for k = 1 : size(gtBoxes,1)
            gtBoxes = double(gtBoxes);
            rectangle('Position',gtBoxes(k,1:4), 'EdgeColor', 'b');
            text(gtBoxes(k,1), gtBoxes(k,2)-5, num2str(k), 'FontSize', 10, 'Color', 'b');
        end
        title gtSeg;
        
        subplot(2,3,3);
        imshow(segFCN-1,cMap);
        for k = 1 : size(boxes,1)
            boxes = double(boxes);
            rectangle('Position',boxes(k,1:4), 'EdgeColor','r');
            text(boxes(k,1), boxes(k,2)-5, num2str(k), 'FontSize', 10, 'Color', 'r');
        end
        for k = 1 : size(gtBoxes,1)
            gtBoxes = double(gtBoxes);
            rectangle('Position',gtBoxes(k,1:4), 'EdgeColor', 'b');
            text(gtBoxes(k,1), gtBoxes(k,2)-5, num2str(k), 'FontSize', 10, 'Color', 'b');
        end
        title fcnSeg;
        
        subplot(2,3,4);
        sz = boxes(j,[4 3]);
        imshow(imresize(patch, sz));
        
        title(['patch' num2str(j)]);
        
        subplot(2,3,5);
        imshow(imresize(gtMask,sz,'nearest'),cMap);
        title(['gtMask' num2str(j)])
        
        subplot(2,3,6);
        imshow(imresize(pMask,sz,'nearest'),cMap);
        title(['pMask' num2str(j)]);
        
        %waitforbuttonpress;

end