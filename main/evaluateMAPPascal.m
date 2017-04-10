function [  ] = evaluateMAPPascal( instances, varargin )

    % Default parameter values
    opts.IoUThreshold = 0.5;
    opts.confidenceLevels = linspace(0.1,1,10);
    opts.debug = false;
    
    % Override with user supplied values
    opts = vl_argparse(opts,varargin);

    % Initialize PASCAL VOC devkit functions
    VOCinit;
    cat_names = {'background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat', ...
    'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant', ...
    'sheep','sofa','train','tvmonitor'};

    % Other initializations
    truePositives = zeros(numel(opts.confidenceLevels),numel(cat_names)-1);
    falsePositives = zeros(numel(opts.confidenceLevels),numel(cat_names)-1);
    falseNegatives = zeros(numel(opts.confidenceLevels),numel(cat_names)-1);

    % Find all image ids
    imgIds = unique({instances.imgId});
    
    % For each image
    for iImg = 1 : numel(imgIds)      
        
        % Select all predicted instances for the current image
        currentInstances = instances(ismember({instances.imgId},imgIds{iImg}));
        
        % Load annotation and segmentation
        annopath = sprintf(VOCopts.annopath,imgIds{iImg});
        objsegpath = sprintf(VOCopts.seg.instimgpath,imgIds{iImg});
        ann = PASreadrecord(annopath);
        objseg = imread(objsegpath);       
        
        % Find all ground truth instances
        gtInstances = struct();
        for iGt = 1 : numel(ann.objects)
            gtInstances(iGt).mask = (objseg == iGt) ;
            gtInstances(iGt).class = ann.objects(iGt).class;
        end
        
        % For each confidence level threshold
        for iConfThreshold = 1 : numel(opts.confidenceLevels)
            
            confThreshold = opts.confidenceLevels(iConfThreshold);
            
            % Select only confident enough instances
            confidentInstances = currentInstances([currentInstances.score] > confThreshold);
            
            % Plot debug information      
            if opts.debug
                cMap = distinguishable_colors(21, [0 0 0]);
                multiImg1 = zeros(size(objseg,1), size(objseg,2), numel(gtInstances));
                multiImg2 = zeros(size(objseg,1), size(objseg,2), numel(confidentInstances));
                for i = 1 : numel(gtInstances)
                    multiImg1(:,:,i) = gtInstances(i).mask;                                    
                end
                for i = 1 : numel(confidentInstances)
                    multiImg2(:,:,i) = confidentInstances(i).segmentation;
                end
                I1 = createOverlayFromMultiImg(multiImg1);
                I2 = createOverlayFromMultiImg(multiImg2);
                imshow(imfuse(I1,I2,'method','blend')); hold on;
                
                % Place markers on centers of ground truth instances
                for i = 1 : numel(gtInstances)
                    center = findCenterBinaryMask(gtInstances(i).mask);
                    plot(center(2),center(1),'--gs',...
                        'LineWidth',2,...
                        'MarkerSize',10,...
                        'MarkerEdgeColor','g',...
                        'MarkerFaceColor',cMap(find(strcmp(cat_names,gtInstances(i).class)),:));
                    text(center(2)+5,center(1),[num2str(i) '-' gtInstances(i).class], 'Color', 'g');
                end
                
                % Place markers on centers of predicted instances
                for i = 1 : numel(confidentInstances)
                    center = findCenterBinaryMask(confidentInstances(i).segmentation);
                    scatter(center(2),center(1),60, 'filled', ...
                        'MarkerFaceAlpha',2/3, ...
                        'MarkerEdgeColor','r', ...
                        'MarkerFaceColor',cMap(confidentInstances(i).catId,:));
                    text(center(2),center(1)+5,[num2str(i) '-' cat_names{confidentInstances(i).catId}], 'Color', 'r');
                end
                
                title(['IMG: ' imgIds{iImg} '    conf: ' num2str(confThreshold)]);
                drawnow;
                hold off;
            end
            
            
            % For each class
            for class = 2 : 21
                
                % Get class name
                className = cat_names(class);
               
                % Select predicted and ground truth instances of the given
                % class
                classConfidentInstances = confidentInstances( [confidentInstances.catId] == class ) ;
                classGtInstances = gtInstances( ismember({gtInstances.class},className) ) ;
                
                % Display debug message and initialize index mapping to
                % ground truth
                if opts.debug
                    disp(['-------- Analyzing class:' className{1} ' --------']);
                    gtIdxs = 1:numel(classGtInstances);
                end
                
                % For each predicted instance
                for iInst = 1 : numel(classConfidentInstances)                      
                        
                    % Find the ground truth with the highest IoU
                    highestIoU = 0;
                    idxHighestIoU = 0;
                    for iGt = 1 : numel(classGtInstances)
                        IoU = computeIoU( classConfidentInstances(iInst).segmentation , classGtInstances(iGt).mask, objseg == 255 );
                        if IoU > highestIoU
                            highestIoU = IoU;
                            idxHighestIoU = iGt;
                        end
                    end
                    
                    % Display debug message
                    if opts.debug
                        disp(['analyzing instance ' num2str(iInst)]);
                        if idxHighestIoU ~= 0
                            disp(['Matches with gt no. ' num2str(gtIdxs(idxHighestIoU)) ' - IoU: ' num2str(highestIoU)]);
                        end
                    end


                    % If IoU good enough, add true positive and remove the matched
                    % ground truth 
                    if highestIoU > opts.IoUThreshold
                        truePositives(iConfThreshold, class-1) = truePositives(iConfThreshold, class-1) + 1;
                        classGtInstances(idxHighestIoU) = [];
                        if opts.debug
                            disp('true positive'); 
                            gtIdxs(idxHighestIoU) = [];
                        end
                        
                    % Otherwise add false positive
                    else
                        falsePositives(iConfThreshold, class-1) = falsePositives(iConfThreshold, class-1) + 1;
                        if opts.debug, disp('false positive'); end
                    end
 
                end
                
                % After all predicted instances have been processed, the
                % remaining ground truth instances are false negatives
                falseNegatives(iConfThreshold, class-1) = falseNegatives(iConfThreshold, class-1) + numel(classGtInstances);
                if opts.debug && numel(classGtInstances) > 0
                    disp([num2str(numel(classGtInstances)) ' gt were not found. Adding to false negatives']);
                end
             
            end
            if opts.debug
                waitforbuttonpress;
                clc;
            end
        end
        


        
    end
    
end


function IoU = computeIoU(instance, groundTruth, ignoreLabels)
    
     instance(ignoreLabels) = 0;
     I = instance & groundTruth;
     U = instance | groundTruth;
     IoU = sum(I(:))/sum(U(:));

end

function center = findCenterBinaryMask(mask)

    [x,y] = find(mask == 1) ;
    center = [mean(x) mean(y)] ;

end

