function [ instances ] = run_full_net( imgs, imgIds, varargin )
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
    
    % Run FCN-8s in all images
    if opts.verbose
        fprintf(' (%.3fs)',toc);
        tic;
        fprintf('\nrunning fcn-8s...');
    end    
    segFCN = run_fcn_8s(imgs, 'gpu', opts.gpu);

    if opts.verbose   
        fprintf(' (%.3fs)',toc);
        tic;
        fprintf('\nrunning masknet...');
    end 
    
    % Load masknet
    net = masknet3_init({'preInitModelPath', opts.masknetPath},{'batchSize', 1});
    net.mode = 'test';
    if opts.gpu >= 1
        net.move('gpu');
    end
    
    % For each image
    for i = 1 : numel(imgs)
        
        % For each detection
        for j = 1 : size(detections{i},1)
            
            % Generate the image patch corresponding to the current bbox
            patch = cutPatch(imgs{i},detections{i}(j,1:4));
            patch = imresize(patch, masknetInputSize);
    
            % Generate the partial mask corresponding to the current bbox
            pMask = generatePartialMask(detections{i}, j, segFCN{i}, masknetInputSize); 
            
            % Put the partial mask in the format expected by matconvnet
            pMask(pMask == 0) = -1;         
            
            % Transfer input data to gpu if needed
            if opts.gpu >= 1
                inputs = {'input',gpuArray(single(patch)),'pmask',gpuArray(single(pMask))};
            else
                inputs = {'input',single(patch),'pmask',single(pMask)};
            end
            
            % Run masknet
            net.eval(inputs);
            mask = gather(net.vars(net.getVarIndex('prediction')).value);
            mask = single(mask > 0);
            
            % Plot debug info
            if opts.debug
                plotDebugInfo(imgs{i}, detections{i}, segFCN{i}, j, patch, pMask, mask);
            end
            
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
        
    end
    
    if opts.verbose
        fprintf(' (%.3fs)\n',toc);
    end
            


end

function plotDebugInfo(image, detections, sseg, j, patch, pMask, mask)

    % Show original image + bboxes
    subplot(3,6,[1:3 7:9]);
    imshow(image);
    plotBboxes(detections,j);
    title 'Original image';
    % add titles, add axis images
    
    % Show segmentation map + bboxes    
    subplot(3,6,[4:6 10:12]);
    imagesc(sseg+1); 
    axis image;
    plotBboxes(detections,j);
    title 'Semantic segmentation + bboxes';
    
    % Show patch
    subplot(3,6,[13 14]);  
    imshow(patch);
    axis image;
    title 'Patch';
    
    % Show partial mask
    subplot(3,6,[15 16]);
    imagesc(pMask);
    axis image;
    title 'Partial mask';
    
    % Show generated instance
    subplot(3,6,[17 18]);
    imagesc(mask);
    axis image;
    title 'Instance';
    
    waitforbuttonpress;

end

function plotBboxes(boxes, j)
% Plot all bounding boxes specified in 'boxes', along with their confidence
% level. The 'j'-th box is plotted in a different color.
    
    for k = 1 : size(boxes,1)
            boxes = double(boxes);
            if k ~= j
                rectangle('Position',boxes(k,1:4), 'EdgeColor','r');
                text(boxes(k,1), boxes(k,2)-5, num2str(boxes(k,end-1)), 'FontSize', 10, 'Color', 'r');
            else
                rectangle('Position',boxes(k,1:4), 'EdgeColor','g');
                text(boxes(k,1), boxes(k,2)-5, num2str(boxes(k,end-1)), 'FontSize', 10, 'Color', 'g');
            end            
    end
    
end


