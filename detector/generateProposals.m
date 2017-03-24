function [ boxes ] = generateProposals( im )

    colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
    colorType = colorTypes{1}; % Single color space for demo

    % Here you specify which similarity functions to use in merging
    simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
    simFunctionHandles = simFunctionHandles(1:4); % Two different merging strategies

    % Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
    % Note that by default, we set minSize = k, and sigma = 0.8.
    k = 50; % controls size of segments of initial segmentation. 
    minSize = k;
    sigma = 0.3; %0.8

    % Perform Selective Search
    [boxes, ~ , ~, ~] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);
    
    % Transform the boxes to fast-rcnn convention 
    boxes = boxes(:,[2 1 4 3]);
    boxes = boxes - 1 ;
    

end

