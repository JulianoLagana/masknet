function [ gtMask ] = generateGtMask( predictedBox, gtBoxes, ann, objseg, gtMaskSize )
% generateGtMask(predictedBox, gtBoxes, ann, objseg, gtMaskSize) generates
% the ground truth data to train masknet. This is used to generate training
% examples from the PASCAL VOC dataset.
%
% Inputs:
%
%   - predictedBox : vector specifying the bounding box to be used. This vector is 
%   specified as [x y w h], where x and y are the coordinates of the top 
%   left corner of the bounding box and w and h specify the height and 
%   width of it. Note that the coordinate system used for x and y is 
%   1-based, meaning that the topmost left pixel of any image has
%   coordinates (1,1) (not 0,0).
%
%   - gtBoxes : Nx4 matrix specifying the N ground truth bounding boxes 
%   present in the image being considered. Each bounding box must follow
%   the same conventions specified for the input 'predictedBox'.
%
%   - ann : annotation data structure provided by PASCAL VOC for the image
%   being considered.
%
%   - objseg : ground truth segmentation for the image being considered
%
%   - gtMaskSize : desired spatial dimensions for the generated ground
%   truth data.
%
% Outputs:
%
%   - gtMask : Binary mask containing pixels that belong to {0, 1, 2}. 0
%   means false, 1 means true and 2 means ignore.
%

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
        gtMask = int8(objseg == mapObjIdxs(IoUidx)) + int8(objseg == 255)*2;

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
    gtMask = int8(gtMask);

    % Resize it to the desired size
    gtMask = imresize(gtMask, gtMaskSize, 'nearest');
    
end

