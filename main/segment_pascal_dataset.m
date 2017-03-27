VOCinit;

% Parameters
opts.savePath = 'data/VOC2012/SegmentationFCN';
opts.saveSize = [224 224];
DEBUG = true;

% Read all image ids in the training and validation Pascal VOC dataset
imgset = 'trainval';
ids = textread(sprintf(VOCopts.seg.imgsetpath,imgset),'%s');

% For each image
for i = 1 : numel(ids)
    
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
    
    % Get ground truth bounding boxes and put them in MATLAB's convention 
    % (i.e. [x1,y1,x2,y2]). Note: PASCAL bboxes are 1-based.
    gtBoxes = reshape([ann.objects.bbox],4,[])';
    for iBox = 1 : size(gtBoxes,1)
        gtBoxes(iBox,3) = gtBoxes(iBox,3)-gtBoxes(iBox,1);
        gtBoxes(iBox,4) = gtBoxes(iBox,4)-gtBoxes(iBox,2);
    end
    
    % Load FCN-8s
    net = dagnn.DagNN.loadobj(load('data/models/pascal-fcn8s-dag.mat')) ;
    
    % Segment the image using FCN-8s
    segFCN = segment(net, img);
    sz = size(img);
    sz = sz(1:2);
    segFCN = imresize(segFCN,sz,'nearest');
    segFCN = uint8(segFCN);
    clear net;
    
    % Find bboxes using Fast-rcnn. Note: Fast-rcnn boxes are 0-based.
    props = generateProposals(img); % this outputs 0-based bboxes
    boxes = run_fast_rcnn(img,props);
    
    % Put bboxes in MATLAB's convention.
    for iBox = 1 : size(boxes,1)
        boxes(iBox,3) = boxes(iBox,3)-boxes(iBox,1);
        boxes(iBox,4) = boxes(iBox,4)-boxes(iBox,2);
    end
    boxes(:,1:4) = boxes(:,1:4)+1;

    % For each bbox found
    for j = 1 : size(boxes,1)
        
        % Generate image patch corresponding to the current bbox
        patch = cutPatch(img,boxes(j,1:4));
        
        % Generate the corresponding ground truth to the current bbox
        gtMask = generateGtMask(boxes(j,:), gtBoxes, ann, objseg, opts.saveSize);        
        
        % Generate the corresponding partial mask to the current bbox
        pMask = generatePartialMask(boxes, j, segFCN, opts.saveSize); 
        
        % DEBUG
        if(DEBUG)
            plotDebugInfo(img,clsseg,segFCN,boxes,gtBoxes,patch,pMask,gtMask,j);
        end
        
        % Put the masks in the format expected by matconvnet
        
    end
    
end


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
        imshow(patch);
        sz = size(patch);
        sz = sz(1:2);
        title(['patch' num2str(j)]);
        
        subplot(2,3,5);
        imshow(imresize(gtMask,sz,'nearest'),cMap);
        title(['gtMask' num2str(j)])
        
        subplot(2,3,6);
        imshow(imresize(pMask,sz,'nearest'),cMap);
        title(['pMask' num2str(j)]);
        
        waitforbuttonpress;

end