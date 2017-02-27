function [ net ] = load_vgg_feature_computer( vgg_path )

    net = load(vgg_path) ;
    net = vl_simplenn_tidy(net) ;
    
    % Remove all layers after the 14th one (7 layers removed in total)
    for i = 1 : 7
        net.layers(15) = [];
    end
 
end

