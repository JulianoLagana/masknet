function detections = run_fast_rcnn(imgs, boxMatrix )
%run_fast_rcnn(im, boxes) Runs the Fast-RCNN net using as input the
%provided images inside the cell array 'im' and the bounding box proposals 
%inside the cell array 'boxes'.
%
% Inputs:
%
%   - im : Cell array, each entry is one input image.
%
%   - boxes : Cell array, each entry is a Nx4 matrix specifying the N 
%   bounding boxes initial proposals. Each bounding box is specified as a 
%   1x4 vector [x y w h], where x and y are the coordinates of the top left
%   corner of the bounding box and w and h specify the height and width of 
%   it. Note that the coordinate system used for x and y is 0-based, 
%   meaning that the topmost left pixel of any image has coordinates (0,0) 
%   (not 1,1).
%
%
% This was created from a file that was part of the VLFeat library and was 
% made available under the terms of the BSD license of the VLFeaet library
% (see the COPYING file of such library).

% Check that there are the same number of images and cells in the boxMatrix
assert(numel(imgs) == numel(boxMatrix));

addpath(fullfile(vl_rootnn,'examples','fast_rcnn','bbox_functions')) ;

opts.modelPath = '' ;
opts.gpu = 1 ;
opts.confThreshold = 0.8 ;
opts.nmsThreshold = 0.3 ;
opts.modelPath = 'data/models/fast-rcnn-vgg16-pascal07-dagnn.mat' ;

% Load the network and put it in test mode.
net = load(opts.modelPath) ;
net = dagnn.DagNN.loadobj(net);
net.mode = 'test' ;

% Mark class and bounding box predictions as `precious` so they are
% not optimized away during evaluation.
net.vars(net.getVarIndex('cls_prob')).precious = 1 ;
net.vars(net.getVarIndex('bbox_pred')).precious = 1 ;

% Evaluate network either on CPU or GPU.
if numel(opts.gpu) > 0
  gpuDevice(opts.gpu) ;
  net.move('gpu') ;
end

for i = 1 : numel(imgs)

    % Cast image and candidate bounding boxes.
    im = single(imgs{i}) ;
    imo = imgs{i}; % keep original image
    boxes = single(boxMatrix{i}') + 1 ;
    boxeso = boxMatrix{i} - 1; % keep original boxes

    % Resize images and boxes to a size compatible with the network.
    imageSize = size(im) ;
    fullImageSize = net.meta.normalization.imageSize(1) ...
        / net.meta.normalization.cropSize ;
    scale = max(fullImageSize ./ imageSize(1:2)) ;
    im = imresize(im, scale, ...
                  net.meta.normalization.interpolation, ...
                  'antialiasing', false) ;
    boxes = bsxfun(@times, boxes - 1, scale) + 1 ;

    % Remove the average color from the input image.
    imNorm = bsxfun(@minus, im, net.meta.normalization.averageImage) ;

    % Convert boxes into ROIs by prepending the image index.
    rois = [ones(1,size(boxes,2)) ; boxes] ;

    % Evaluate network either on CPU or GPU.
    if numel(opts.gpu) > 0
      imNorm = gpuArray(imNorm) ;
      rois = gpuArray(rois) ;
    end

    net.conserveMemory = false ;
    net.eval({'data', imNorm, 'rois', rois});

    % Extract class probabilities and  bounding box refinements
    probs = squeeze(gather(net.vars(net.getVarIndex('cls_prob')).value)) ;
    deltas = squeeze(gather(net.vars(net.getVarIndex('bbox_pred')).value)) ;

    % Save the best results for each category (except background = 1)
    detections{i} = [];
    for c = 2:numel(net.meta.classes.name)
      cprobs = probs(c,:) ;
      cdeltas = deltas(4*(c-1)+(1:4),:)' ;
      cboxes = bbox_transform_inv(boxeso, cdeltas);
      cls_dets = [cboxes cprobs'] ;

      keep = bbox_nms(cls_dets, opts.nmsThreshold) ;
      cls_dets = cls_dets(keep, :) ;

      sel_boxes = find(cls_dets(:,end) >= opts.confThreshold) ;
      c_detections = cls_dets(sel_boxes,:);
      detections{i} = [detections{i}; [c_detections repmat(c,size(c_detections,1),1)]];

    end


end

end