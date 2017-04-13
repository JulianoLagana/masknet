function [ instances ] = run_full_net_useGtMasks( imgs, imgIds, anns, objsegs, varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    % Default parameters
    opts.verbose = false;
    opts.debug = false;
    opts.masknetPath = 'data\experiments\masknet3\VOC2012\pascal_imdb\lr2e-06_wd0_mom0p9_batch50_maskSize224  224_M224_f200/net-epoch-20';
    opts.initInstances = [];
    opts.gpu = 1;
    opts.debug = false; 
    
    % Override default parameters with user-supplied values
    opts = vl_argparse(opts,varargin);
    
    % Initializations
    masknetInputSize = [224 224];
    instances = opts.initInstances;

    % Generate proposals for all images 
    if opts.verbose
        tic;
        fprintf('generating proposals...');
    end
    for i = 1 : numel(imgs)  
       props{i} = generateProposals(imgs{i});
    end
    
    % Run fast-rcnn
    if opts.verbose
        fprintf(' (%.3fs)',toc);
        tic;
        fprintf('\nrunning fast-rcnn...');
    end    
    detections = run_fast_rcnn(imgs,props,'confThreshold',0.1,'gpu',opts.gpu);

    % Put detections in MATLAB's bbox convention.
    for iImage = 1 : numel(detections)
        for iBox = 1 : size(detections{iImage},1)
          detections{iImage}(iBox,3) = detections{iImage}(iBox,3)-detections{iImage}(iBox,1);
          detections{iImage}(iBox,4) = detections{iImage}(iBox,4)-detections{iImage}(iBox,2);
        end
    detections{iImage}(:,1:4) = detections{iImage}(:,1:4)+1;   
    end    

    if opts.verbose   
        fprintf(' (%.3fs)',toc);
        tic;
        fprintf('\nUsing ground truth as masks...');
    end 
    
    % For each image
    for i = 1 : numel(imgs)       
            
        gtBoxes = reshape([anns{i}.objects.bbox],4,[])';
        for iBox = 1 : size(gtBoxes,1)
            gtBoxes(iBox,3) = gtBoxes(iBox,3)-gtBoxes(iBox,1);
            gtBoxes(iBox,4) = gtBoxes(iBox,4)-gtBoxes(iBox,2);
        end
        
        % For each detection
        for j = 1 : size(detections{i},1)
                        
            % Use ground truth mask of the highest IoU gt bbox 
            mask = generateGtMask(detections{i}(j,:), gtBoxes, anns{i}, objsegs{i}, masknetInputSize);
            mask(mask == 2) = 0;
            
            % Resize the mask to the initial bounding box size
            bbox = round(detections{i});                                    
            x = max(bbox(j,1), 1);
            y = max(bbox(j,2), 1);
            w = bbox(j,3);
            h = bbox(j,4);
            s = size(imgs{i});
            mask = imresize(mask, [numel(y:min(y+h,s(1))) , numel(x:min(x+w,s(2)))], 'nearest');
            
            % Create segmentation map for this instance
            segImage = zeros(size(imgs{i},1), size(imgs{i},2));
            segImage(y:min(y+h,s(1)) , x:min(x+w,s(2))) = mask;  
            
            % Add instance to found instances
            nInstances = numel(instances);
            instances(nInstances+1).imgId = imgIds{i};
            instances(nInstances+1).segmentation = segImage;
            instances(nInstances+1).score = detections{i}(j,end-1);
            instances(nInstances+1).catId = detections{i}(j,end);
            
        end    
        
        if opts.debug
            plotDebug(i,imgs,gtBoxes,imgIds, instances, detections{i});
        end
        
    end
    
    if opts.verbose
        fprintf(' (%.3fs)\n',toc);
    end
            


end


function plotDebug(i, imgs, gtBoxes, imgIds, instances, detections)

    clf;
    subplot(1,2,1) , imshow(imgs{i})
    hold on;
    for k = 1 : size(gtBoxes,1)
        gtBoxes = double(gtBoxes);
        rectangle('Position',gtBoxes(k,1:4), 'EdgeColor', 'b');
        text(gtBoxes(k,1), gtBoxes(k,2)-5, num2str(k), 'FontSize', 10, 'Color', 'b');
    end
    
    for k = 1 : size(detections,1)
        detections = double(detections);
        rectangle('Position',detections(k,1:4), 'EdgeColor', 'r');
        text(detections(k,1), detections(k,2)-5, num2str(k), 'FontSize', 10, 'Color', 'r');
    end

    inst = instances(strcmp({instances.imgId},imgIds{i}));
    for i = 1 : numel(inst)
        multiImg(:,:,i) = inst(i).segmentation;
    end

    subplot(1,2,2)
    imshow(createOverlayFromMultiImg(multiImg))
    waitforbuttonpress;

end