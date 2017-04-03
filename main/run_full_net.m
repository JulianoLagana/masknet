function [  ] = run_full_net( imgs, varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    % Default parameters
    opts.verbose = false;
    opts.masknetPath = 'data\experiments\masknet3\VOC2012\pascal_imdb\lr2e-06_wd0_mom0p9_batch50_maskSize224  224_M224_f200/net-epoch-20';
    
    % Override default parameters with user-supplied values
    opts = vl_argparse(opts,varargin);
    
    % Initializations
    masknetInputSize = [224 224];

    % Generate proposals for all images 
    if opts.verbose
        disp('generating proposals...');
    end
    for i = 1 : numel(imgs)  
       props{i} = generateProposals(imgs{i});
    end
    
    % Run fast-rcnn
    if opts.verbose
        disp('running fast-rcnn...');
    end    
    detections = run_fast_rcnn(imgs,props);

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
        disp('running masknet...');
    end    
    segFCN = run_fcn_8s(imgs, 'gpu', 1);
    
    % Load masknet
    net = loadMasknet(opts.masknetPath,'batchSize',1);
    
    % For each image
    for i = 1 : numel(imgs)
        
        segImage = zeros(size(imgs{i},1), size(imgs{i},2));
        instanceNumber = 1;
        
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
            
            % Add this instance to segImage
            x = max(bbox(j,1), 1);
            y = max(bbox(j,2), 1);
            w = bbox(j,3);
            h = bbox(j,4);
            s = inf*size(imgs{i});
            segImage(y:min(y+h,s(1)) , x:min(x+w,s(2))) = mask.*instanceNumber;
            
            instanceNumber = instanceNumber + 1;
            
        end
        
        % Show instance segmentation
        subplot(1,2,1); 
        imshow(imgs{i})
        subplot(1,2,2);
        imagesc(segImage);
        axis image;
        waitforbuttonpress;
        
    end
            


end

