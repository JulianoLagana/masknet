function showOutputMaskproposalMasknet(varargin)

    % Initialize PASCAL VOC devkit functions
    VOCinit;
    cat_names = {'background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat', ...
    'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant', ...
    'sheep','sofa','train','tvmonitor'};

    % Parameters
    opts.gpu = true;
    opts.batchSize = 100;
    opts.masknetPath = 'data\experiments\masknet3\VOC2012\pascal_imdb\lr1e-06_wd5e-05_mom0p9_batch40_preInitModelPathdata!experiments!masknet3!COCO_datasets!centered_imdb!lr1e-06_wd5e-05_mom0p9_batch20_M224_f300!net-epoch-16pmat/net-epoch-15.mat';
    opts.deepmaskPath = 'data/experiments/deepmask_dag\VOC2012\pascal_imdb\lr1e-06_wd5e-05_mom0p9_batch40_preInitModelPathdata!experiments!deepmask_dag!COCO_datasets!centered_imdb!lr1e-06_wd5e-05_mom0p9_batch40!net-epoch-3pmat/net-epoch-8.mat';
    opts.fastrcnnThreshold = 0.7;
    opts.verbose = true;
    opts.debug = false;     
    opts.saveDir = 'data/output/mask_proposal/';
    opts = vl_argparse(opts,varargin);

    % Other initializations
    batchNumber = 1;
    subplot = @(m,n,p) subtightplot (m, n, p, [0 0.002], [0. 0.], [0. 0.0]);
    masknetInputSize = [224 224];
    mkdir(opts.saveDir);
    figure('visible','off');

    % Choose first 100 validation images
    ids = textread(sprintf(VOCopts.seg.imgsetpath,'val'),'%s');
    nImages = numel(ids);    

    while ~isempty(ids)

        batchSize = min(opts.batchSize,numel(ids));

        idsToProcess = ids(1:batchSize);
        ids(1:batchSize) = [];

        fprintf('------------- Batch %d/%d -------------\n', batchNumber, ceil(nImages/opts.batchSize));

        % Load the images in a cell array
        imgs = cell(1,batchSize);
        for i = 1 : batchSize
            imgpath = sprintf(VOCopts.imgpath,idsToProcess{i});
            imgs{i} = imread(imgpath);
        end

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
        detections = run_fast_rcnn(imgs,props,'confThreshold',opts.fastrcnnThreshold,'gpu',opts.gpu);

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
            fprintf('\nrunning masknet version with partial masks...');
        end 

        % Load masknet with partial masks
        net = masknet3_init({'preInitModelPath', opts.masknetPath},{'batchSize', 1});
        net.mode = 'test';
        if opts.gpu >= 1
            net.move('gpu');
        end

        % Process all detections with Masknet with partial masks
        nProcessed = 0;
        for i = 1 : numel(imgs)

            % For each detection
            for j = 1 : size(detections{i},1)

                % Generate the image patch corresponding to the current bbox
                patch0 = cutPatch(imgs{i},detections{i}(j,1:4));
                patch = imresize(patch0, masknetInputSize);

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
                mask(mask == 0) = -1;

                % Add output to processed detections
                nProcessed = nProcessed + 1;
                [ss(1), ss(2), ~] = size(patch0);
                patches{nProcessed} = patch0;
                partialMasks{nProcessed} = imresize(pMask,ss,'nearest');
                processedDetectionsWith{nProcessed} = imresize(mask,ss,'nearest');                

            end        

        end
        
        clear net;

        if opts.verbose
            fprintf(' (%.3fs)',toc);
            tic;
            fprintf('\nrunning masknet version without partial masks...');
        end 
        
        % Load deepmask
        net = deepmask_dag_init({'preInitModelPath', opts.deepmaskPath},{'batchSize', 1});
        net.mode = 'test';
        if opts.gpu >= 1
            net.move('gpu');
        end
        
        % Process all detections with Masknet without partial masks
        nProcessed = 0;
        for i = 1 : numel(imgs)
        
            % For each detection
            for j = 1 : size(detections{i},1)

                % Generate the image patch corresponding to the current bbox
                patch0 = cutPatch(imgs{i},detections{i}(j,1:4));
                patch = imresize(patch0, masknetInputSize);      

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

                % Add output to processed detections
                nProcessed = nProcessed + 1;
                [ss(1), ss(2), ~] = size(patch0);
                processedDetectionsWithout{nProcessed} = imresize(mask,ss,'nearest');
                
            end        
        
        end
       
        if opts.verbose
            fprintf(' (%.3fs)',toc);
        end 
        
        % Plot results
        for i = 1 : nProcessed

            subplot(1,3,1);
            imshow(patches{i});
            axis normal fill;
            
            subplot(1,3,2);
            imagesc(partialMasks{i});
            axis off
            colormap gray;
            
            subplot(1,3,3);
            imagesc(processedDetectionsWith{i});
            axis off;
            colormap gray;
            
            ss = get(0,'screensize');
            [s1, s2, ~] = size(patches{i});
            set(gcf,'units','pixels');
            set(gcf,'Position',[50 ss(4)-s1*1.5-100 s2*3*1.2 s1*1.2])
            
            saveas(gcf,[opts.saveDir 'val_detection' num2str(i) '.jpg']);
        end
        

    end        
         
end