function [  ] = show_net_performance( net , imdb, start)

    % Remove the loss layer from the net
    net.layers(end) = [];

    % Pass the net to the gpu
    net = vl_simplenn_move(net, 'gpu') ;
    
    % Open imdb file
    file = matfile(imdb);
    nImages = getfield(whos(file),'size');  nImages = nImages(4);
    
    % Initializa fullscreen figure
    figure('units','normalized','outerposition',[0 0 1 1]);
    
    for i = round(nImages*start)+1 : nImages
        
        % Load the image and the mask
        imo = file.imdb(:,:,:,i);
        im = single(imresize(imo, [224 224]));
        im = gpuArray(im);
        mask = file.masks(:,:,:,i);
        
        % Process the feature map using the provided network
        res = vl_simplenn(net,im);
        
        % Plot the output overlaid on the orignal image
        subplot(1,2,1);
        out = res(end).x;
        out = out(:,:,:,1);
        out = gather(out);
        image(imfuse(imo,out>0,'blend','Scaling','joint'));
        axis image;
        
        % Plot the ground truth overlaid on the original image
        subplot(1,2,2);
        image(imfuse(imo,mask,'blend','Scaling','joint'));
        axis image;
        
        % Wait for user visualization
        waitforbuttonpress;
        
    end
        

end

