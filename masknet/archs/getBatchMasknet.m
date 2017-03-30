
function inputs = getBatchMasknet(opts, imdb, batch)

    images = single(imdb.imdb(:,:,:,batch));
    
    partial_masks = single(imdb.partial_masks(:,:,1,batch));
    partial_masks(partial_masks == 0) = -1;
    
    masks = single(imdb.masks(:,:,1,batch));
    masks(masks == 0) = -1;
    
    if numel(opts.gpus) > 0
        images = gpuArray(images);
        partial_masks = gpuArray(partial_masks);    
        masks = gpuArray(masks);  
    end
    
    inputs = {'input',images,'pmask',partial_masks,'gtMask',masks};
    
end