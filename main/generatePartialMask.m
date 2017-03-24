function [ pMask ] = generatePartialMask( boxes, j, segmentation, pMaskSize)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    % Filter the segmentation to keep only the pixels of the class of
    % the chosen predicted bbox
    predictedClass = boxes(j,end);
    mask1 = segmentation == predictedClass;

    % For all other bounding boxes of the same class, occlude the mask
    % with them
    matchingIdxs = find(boxes(:,end)==predictedClass)';
    matchingIdxs(matchingIdxs == j) = [];
    mask2 = mask1;
    for iBox = matchingIdxs
        mask2 = occludeMask(mask2,boxes(iBox,1:4));
    end

    % Crop segmentation to current bbox
    mask3 = cutPatch(mask2,boxes(j,1:4));
    
    % Resize and convert to uint8
    pMask = uint8(imresize(mask3,pMaskSize));
    
    

end

