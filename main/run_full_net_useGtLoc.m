function [ instances ] = run_full_net_useGtLoc( imgs, imgIds, anns, varargin )

    % Default parameters
    opts.verbose = false;
    opts.debug = false;
    opts.masknetPath = 'data\experiments\masknet3\VOC2012\pascal_imdb\lr2e-06_wd0_mom0p9_batch50_maskSize224  224_M224_f200/net-epoch-20';
    opts.initInstances = [];
    opts.gpu = 1;
    
    % Override default parameters with user-supplied values
    opts = vl_argparse(opts,varargin);
    
    % Initializations
    masknetInputSize = [224 224];
    instances = opts.initInstances;  
    
    if opts.verbose
        tic;
        fprintf('\nloading ground truth data for detections...');
    end   
    
    cat_names = {'background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat', ...
    'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant', ...
    'sheep','sofa','train','tvmonitor'};
    
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