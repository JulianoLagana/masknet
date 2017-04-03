function inputs = getBatchDeepmaskDag(opts, imdb, batch)

    images = single(imdb.imdb(:,:,:,batch));
    masks = single(imdb.masks(:,:,1,batch));
    
    if numel(opts.gpus) > 0
        images = gpuArray(images);   
        masks = gpuArray(masks);  
    end
    
    inputs = {'input',images,'gtMask',masks};
    
end