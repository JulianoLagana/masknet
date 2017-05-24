function [  ] = generate_comparison_merging_strategies(  )
%GENERATE_COMPARISON_MERGING_STRATEGIES Summary of this function goes here
%   Detailed explanation goes here

    %% Generate the names for the merging strategies being compared
    
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
    
    
    %% Plot the IoU for the merging strategies
    
    compare_experiments(expDirs,'error','IoUerr');
    ylabel IoU
    xlabel epochs
    axis([1 20 -Inf Inf]);
    
    % WORKAROUND - Create three false plots to change legend icons (using the same
    % colors used in compare_experiments)
    colors = distinguishable_colors(3,[1 1 1]);
    for i = 1 : 3
        fakeplot{i} = plot(NaN,'o');
        fakeplot{i}.MarkerFaceColor = colors(i,:); fakeplot{i}.MarkerEdgeColor ='none';
    end
    legend([fakeplot{1}, fakeplot{2}, fakeplot{3}], 'Merging strategy 1', 'Merging strategy 2', 'Merging strategy 3', ...
           'Location', 'northwest');

    
    %% Plot the log loss for the merging strategies
    
    compare_experiments(expDirs,'error','objective');
    ylabel Loss
    xlabel epochs
    axis([1 20 0.7e4 3.5e4]);
    
    % WORKAROUND - Create three false plots to change legend icons (using the same
    % colors used in compare_experiments)
    colors = distinguishable_colors(3,[1 1 1]);
    for i = 1 : 3
        fakeplot{i} = plot(NaN,'o');
        fakeplot{i}.MarkerFaceColor = colors(i,:); fakeplot{i}.MarkerEdgeColor ='none';
    end
    legend([fakeplot{1}, fakeplot{2}, fakeplot{3}], 'Merging strategy 1', 'Merging strategy 2', 'Merging strategy 3', ...
        'Location', 'southwest');

end

