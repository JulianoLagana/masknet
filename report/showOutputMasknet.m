function showOutputMasknet(varargin)

    % Initialize PASCAL VOC devkit functions
    VOCinit;
    cat_names = {'background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat', ...
    'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant', ...
    'sheep','sofa','train','tvmonitor'};

    % Parameters
    opts.gpu = true;
    opts.batchSize = 100;
    opts.confidenceLevels = linspace(0,1,100);
    opts.masknetPath = 'data\experiments\masknet3\VOC2012\pascal_imdb\lr1e-06_wd0_mom0p9_batch30_preInitModelPathdata!experiments!masknet3!COCO_datasets!centered_imdb!lr2e-06_wd0_mom0p9_batch30_M224_f300!net-epoch-5pmat/net-epoch-4.mat';
    opts.usesPartialMasks = true;
    opts.fastrcnnThreshold = 0.7;
    opts.debug = false;   
    opts.saveDir = 'data/output/deepmask/';
    opts = vl_argparse(opts,varargin);

    % Other initializations
    batchNumber = 1;
    subplot = @(m,n,p) subtightplot (m, n, p, [0 0], [0 0], [0 0]);
    mkdir(opts.saveDir);
    figure('visible','off');

    % Choose first 100 validation images
    ids = textread(sprintf(VOCopts.seg.imgsetpath,'val'),'%s');
    ids = ids(1:100);
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
        
        % Run full net
        if opts.usesPartialMasks ~= true
           instances = run_full_net_noPartialMasks(imgs,idsToProcess,'verbose', true, 'masknetPath', opts.masknetPath, 'gpu', opts.gpu, 'fastrcnnThreshold', opts.fastrcnnThreshold); 
        else
           instances = run_full_net(imgs, idsToProcess, 'verbose', true, 'masknetPath', opts.masknetPath, 'gpu',opts.gpu, 'fastrcnnThreshold', opts.fastrcnnThreshold, 'debug', opts.debug); 
        end   
        
        
        for i = 1 : numel(idsToProcess)
            
            % Select all predicted instances for the current image
            currentInstances = instances(ismember({instances.imgId},idsToProcess{i}));
           
            % Plot original image in the left
            clf;
            subplot(1,2,1);
            imshow(imgs{i}); 

            % Plot original image in the right
            subplot(1,2,2);
            imshow(imgs{i});
            hold on;   
            
            % Adjust figure size for exactly two images
            ss = get(0,'screensize');
            [s1, s2, ~] = size(imgs{i});
            set(gcf,'units','pixels');
            set(gcf,'Position',[50 ss(4)-s1*1.5-100 s2*2*1.5 s1*1.5])  
            
            % Plot black overlay transparent
            img = zeros(size(imgs{i}));
            hBlack = imshow(img);
            set(hBlack, 'AlphaData', 0.5);
            
            % Plot overlay with all the instance masks
            colors = distinguishable_colors(numel(instances),[1 1 1; 0 0 0.1724]);
            for j = 1 : numel(currentInstances)
                
                % Construct colorized instance
                c = colors(j,:);
                seg = currentInstances(j).segmentation;
                [h,w] = size(seg);
                img = seg.*repmat(permute(c,[3 1 2]), h, w);
                
                % Plot the instance
                hInstance{j} = imshow(img);
                
                % Adjust its alpha map
                alpha = ones(size(seg));
                alpha(seg == 0) = 0;
                set(hInstance{j}, 'AlphaData', alpha*0.7);
                
                % Write its class in the centroid
                [x,y] = find(seg) ;
                t = text(mean(y), mean(x),cat_names(currentInstances(j).catId), ...
                    'HorizontalAlignment','center', 'VerticalAlignment','middle',...
                    'FontSize',20);
                t.BackgroundColor = [1 1 1 0.5];
                t.Margin = 1;
                
            end
            saveas(gcf,[opts.saveDir num2str(idsToProcess{i}) '.jpg'])
            
        end

    end

    
end