function inputs = getBatchDeepmaskDag(opts, imdb, batch)

    images = single(imdb.imdb(:,:,:,batch));
    
    masks = single(imdb.masks(:,:,1,batch));
    masks(masks == 0) = -1;
    
    if numel(opts.gpus) > 0
        images = gpuArray(images);   
        masks = gpuArray(masks);  
    end
    
    inputs = {'input',images,'gtMask',masks};
    
end