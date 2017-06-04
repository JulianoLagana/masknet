function [ instances ] = run_full_net_noPartialMasks_useGtLoc( imgs, imgIds, anns, varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    cat_names = {'background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat', ...
    'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant', ...
    'sheep','sofa','train','tvmonitor'};

    % Default parameters
    opts.verbose = false;
    opts.debug = false;
    opts.masknetPath = 'data\experiments\deepmask_dag\VOC2012\pascal_imdb\lr1e-06_wd5e-05_mom0p9_batch40_preInitModelPathdata!experiments!deepmask_dag!COCO_datasets!centered_imdb!lr1e-06_wd5e-05_mom0p9_batch40!net-epoch-3pmat/net-epoch-8.mat';
    opts.initInstances = [];
    opts.gpu = 1;
    opts.verbose = true;
    opts.fastrcnnThreshold = 0.1;
    
    % Override default parameters with user-supplied values
    opts = vl_argparse(opts,varargin);
    
    if opts.verbose
        fprintf('running masknet...');
        tic;
    end     
    
    % Initializations
    masknetInputSize = [224 224];
    instances = opts.initInstances;

    % Load detections from ground truth bounding boxes
    detections = cell(1,numel(imgs));
    for i = 1 : numel(imgs)
        scores = ones(numel(anns{i}.objects),1);
        detections{i} = [reshape([anns{i}.objects.bbox],4,[])' scores];
        
        for iBox = 1 : size(detections{i},1)
            detections{i}(iBox,3) = detections{i}(iBox,3)-detections{i}(iBox,1);
            detections{i}(iBox,4) = detections{i}(iBox,4)-detections{i}(iBox,2);
            detections{i}(iBox,6) = find(strcmp(anns{i}.objects(iBox).class,cat_names));
        end
    end
    
    % Load deepmask
    net = deepmask_dag_init({'preInitModelPath', opts.masknetPath},{'batchSize', 1});
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
            
            % Transfer input data to gpu if needed
            if opts.gpu >= 1
                inputs = {'input',gpuArray(single(patch))};
            else
                inputs = {'input',single(patch)};
            end
            
            % Run masknet
            net.eval(inputs);
            mask = gather(net.vars(net.getVarIndex('prediction')).value);
            mask = single(mask > 0);
            
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