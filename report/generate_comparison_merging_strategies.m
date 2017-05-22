function [  ] = generate_comparison_merging_strategies(  )
%GENERATE_COMPARISON_MERGING_STRATEGIES Summary of this function goes here
%   Detailed explanation goes here

    archs = {'masknet' 'masknet2' 'masknet3'};
    imdb = 'VOC2012/pascal_imdb';

    % Best found hyperparameters for each
    wd = [5e-5 5e-5 0];
    momentum = [0.9 0.9 0]; 
    learning_rates = [2.1544e-6 4.6416e-6 4.6416e-5 1e-5];

    expDirs = cell(1,3);
    for iArch = 1 : 3
        expDirs{iArch} = [archs{iArch} '/' imdb '/lr' num2str(learning_rates(iArch)) '_wd' num2str(wd(iArch)) '_mom' num2str(momentum(iArch)) '_batch40'];
    end
    expDirs = strrep(expDirs,'.','p');
    compare_experiments(expDirs);

    ylabel IoU
    xlabel epochs
    legend 'Merging strategy 1' 'Merging strategy 2' 'Merging strategy 3';

end

