function [ gtMask ] = generateGtMask( predictedBox, gtBoxes, ann, objseg, gtMaskSize )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    cat_names = {'background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat', ...
    'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant', ...
    'sheep','sofa','train','tvmonitor'};

    % Find gtBoxes of the same class as the predicted box (keep their
    % object number references)
    predictedClass = cat_names{predictedBox(1,end)};
    keep = find(strcmp({ann.objects.class},predictedClass));
    objIdxs = 1:size(gtBoxes,1);
    mapObjIdxs = objIdxs(keep);
    classGtBoxes = gtBoxes(keep,:);
    
    % Find ground truth bbox with higest IoU to this bbox
    [IoU,IoUidx] = max(bboxOverlapRatio(predictedBox(1,1:4),classGtBoxes(:,:)));

    % If any was found
    if ~isempty(IoU) && IoU > 0
        % Generate mask with only the object corresponding to the
        % highest IoU bbox found
        gtMask = uint8(objseg == mapObjIdxs(IoUidx)) + uint8(objseg == 255)*255;

        % Remove ignore labels from other objects that might be inside
        % this region (this does not work perfectly yet)
        temp = occludeMask(gtMask, gtBoxes(mapObjIdxs(IoUidx),:));
        discard = gtMask == temp;
        gtMask(discard) = 0;

        % Cut patch given by the current bbox
        gtMask = cutPatch(gtMask,predictedBox(1,1:4));
    else
        gtMask = 0;
    end 
    gtMask = uint8(gtMask);

    % Resize it to the desired size
    gtMask = imresize(gtMask, gtMaskSize, 'nearest');
    
end

