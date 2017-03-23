function [  ] = visualizeImdb( imdb_path )

    file = matfile(imdb_path);
    figure('units','normalized','outerposition',[0 0 1 1]);
    
    % Find the number of images
    a = whos(file,'imdb');
    nImages = a.size;
    for i = 1 : nImages
        I = file.imdb(:,:,:,i);
        mask = file.masks(:,:,1,i);
        p_mask = file.partial_masks(:,:,1,i);
        
        subplot(1,3,1);
        image(I);
        subplot(1,3,2);
        imagesc(mask);
        subplot(1,3,3);
        imagesc(p_mask);
        
        waitforbuttonpress;
    end


end

