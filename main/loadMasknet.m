function [ net ] = loadMasknet( path, varargin )
%LOADMASKNET Summary of this function goes here
%   Detailed explanation goes here

    opts.batchSize = 10;
    opts = vl_argparse(opts,varargin);

    load(path,'net');
    net = dagnn.DagNN.loadobj(net);
    net.mode = 'test';
    
    % Change the size of the fist bilinear grid
    constantGridGenIdx = net.getLayerIndex('constantGridGen');
    newGrid = repmat(net.layers(constantGridGenIdx).block.value(:,:,:,1),1,1,1,opts.batchSize);
    net.layers(constantGridGenIdx).block.value = newGrid;
    
    % Change the size of the second bilinear grid
    constantGridGenIdx = net.getLayerIndex('constantGridGen2');
    newGrid = repmat(net.layers(constantGridGenIdx).block.value(:,:,:,1),1,1,1,opts.batchSize);
    net.layers(constantGridGenIdx).block.value = newGrid;

end

