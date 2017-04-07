function [ instances ] = run_full_net( imgs, imgIds, varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    % Default parameters
    opts.verbose = false;
    opts.debug = false;
    opts.masknetPath = 'data\experiments\masknet3\VOC2012\pascal_imdb\lr2e-06_wd0_mom0p9_batch50_maskSize224  224_M224_f200/net-epoch-20';
    opts.sensitivity = 0.8;
    opts.initInstances = [];
    
    % Override default parameters with user-supplied values
    opts = vl_argparse(opts,varargin);
    
    % Initializations
    masknetInputSize = [224 224];
    instances = opts.initInstances;

    % Generate proposals for all images 
    if opts.verbose
        tic;
        disp('generating proposals...');
    end
    for i = 1 : numel(imgs)  
       props{i} = generateProposals(imgs{i});
    end
    
    % Run fast-rcnn
    if opts.verbose
        disp('running fast-rcnn...');
    end    
    detections = run_fast_rcnn(imgs,props,'confThreshold',0.2);

    % Put detections in MATLAB's bbox convention.
    for iImage = 1 : numel(detections)
        for iBox = 1 : size(detections{iImage},1)
          detections{iImage}(iBox,3) = detections{iImage}(iBox,3)-detections{iImage}(iBox,1);
          detections{iImage}(iBox,4) = detections{iImage}(iBox,4)-detections{iImage}(iBox,2);
        end
    detections{iImage}(:,1:4) = detections{iImage}(:,1:4)+1;   
    end    
    
    % Run FCN-8s in all images
    if opts.verbose
        disp('running fcn-8s...');
    end    
    segFCN = run_fcn_8s(imgs, 'gpu', 1);

    if opts.verbose   
        toc
        disp('running masknet...');
    end 
    
    % Load masknet
    net = masknet3_init({'preInitModelPath', opts.masknetPath},{'batchSize', 1});
    net.mode = 'test';
    
    % For each image
    for i = 1 : numel(imgs)
        
        segImage = zeros(size(imgs{i},1), size(imgs{i},2));
        
        % For each detection
        for j = 1 : size(detections{i},1)
            
            % Generate the image patch corresponding to the current bbox
            patch = cutPatch(imgs{i},detections{i}(j,1:4));
            patch = imresize(patch, masknetInputSize);
    
            % Generate the partial mask corresponding to the current bbox
            pMask = generatePartialMask(detections{i}, j, segFCN{i}, masknetInputSize); 
            
            % Put the partial mask in the format expected by matconvnet
            pMask(pMask == 0) = -1;         
            
            % Run masknet
            inputs = {'input',single(patch),'pmask',single(pMask)};
            net.eval(inputs);
            mask = net.vars(net.getVarIndex('prediction')).value;
            mask = single(mask > 0);
            
            % Resize the mask to the initial bounding box size
            bbox = round(detections{i});
            mask = imresize(mask, bbox(j,[4 3])+1, 'nearest');
                        
            x = max(bbox(j,1), 1);
            y = max(bbox(j,2), 1);
            w = bbox(j,3);
            h = bbox(j,4);
            s = inf*size(imgs{i});
            
            % Add instance to found instances
            segImage(y:min(y+h,s(1)) , x:min(x+w,s(2))) = mask;  
            nInstances = numel(instances);
            instances(nInstances+1).imgId = imgIds{i};
            instances(nInstances+1).segmentation = segImage;
            instances(nInstances+1).score = detections{i}(j,end-1);
            instances(nInstances+1).catId = detections{i}(j,end);
            
        end        
        
    end
            


end

function plotDebugInfo(image, detections, partSegImage, segMultiImage)

    subplot(2,2,1);
    imshow(image);
    plotBboxes(detections);
    
    subplot(2,2,2);
    imshow(createOverlayFromMultiImg(partSegImage)); % Change this to overlay
    axis image;
    plotBboxes(detections);
    
    subplot(2,2,3);
    imshow(createOverlayFromMultiImg(segMultiImage*0.9)); % Change this to overlay
    axis image;
    plotBboxes(detections);
    
    subplot(2,2,4);
    
    
    waitforbuttonpress;

end

function plotBboxes(boxes)

    cat_names = {'background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat', ...
        'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant', ...
        'sheep','sofa','train','tvmonitor'};
    cMap = VOClabelcolormap(256);
    
    for k = 1 : size(boxes,1)
            boxes = double(boxes);
            rectangle('Position',boxes(k,1:4), 'EdgeColor',cMap(boxes(k,end),:));
            text(boxes(k,1), boxes(k,2)-5, [cat_names{boxes(k,end)} ' ' num2str(boxes(k,end-1))], 'FontSize', 10, 'Color', 'r');
    end
    
end


