function detections = run_fast_rcnn(im, boxes )
%FAST_RCNN_DEMO  Demonstrates Fast-RCNN
%
% Copyright (C) 2016 Abhishek Dutta and Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


addpath(fullfile(vl_rootnn,'examples','fast_rcnn','bbox_functions')) ;

opts.modelPath = '' ;
opts.gpu = 1 ;
opts.confThreshold = 0.5 ;
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

% Load a test image and candidate bounding boxes.
im = single(im) ;
imo = im; % keep original image
boxes = single(boxes') + 1 ;
boxeso = boxes - 1; % keep original boxes

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

% Convert boxes into ROIs by prepending the image index. There is only
% one image in this batch.
rois = [ones(1,size(boxes,2)) ; boxes] ;

% Evaluate network either on CPU or GPU.
if numel(opts.gpu) > 0
  gpuDevice(opts.gpu) ;
  imNorm = gpuArray(imNorm) ;
  rois = gpuArray(rois) ;
  net.move('gpu') ;
end

net.conserveMemory = false ;
net.eval({'data', imNorm, 'rois', rois});

% Extract class probabilities and  bounding box refinements
probs = squeeze(gather(net.vars(net.getVarIndex('cls_prob')).value)) ;
deltas = squeeze(gather(net.vars(net.getVarIndex('bbox_pred')).value)) ;

detections = [];
% Save the best results for each category (except background = 1)
for c = 2:numel(net.meta.classes.name)
  cprobs = probs(c,:) ;
  cdeltas = deltas(4*(c-1)+(1:4),:)' ;
  cboxes = bbox_transform_inv(boxeso', cdeltas);
  cls_dets = [cboxes cprobs'] ;

  keep = bbox_nms(cls_dets, opts.nmsThreshold) ;
  cls_dets = cls_dets(keep, :) ;

  sel_boxes = find(cls_dets(:,end) >= opts.confThreshold) ;
  c_detections = cls_dets(sel_boxes,:);
  detections = [detections; [c_detections repmat(c,size(c_detections,1),1)]];
  
end
