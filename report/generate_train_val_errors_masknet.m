experimentDirs = ...
    {'masknet3\VOC2012\pascal_imdb\lr2e-06_wd0_mom0p9_batch30_M224_f300', ...
    'deepmask_dag\VOC2012\pascal_imdb\lr1e-06_wd5e-05_mom0p9_batch40', ...
    'masknet3\VOC2012\pascal_imdb\lr1e-06_wd0_mom0p9_batch30_preInitModelPathdata!experiments!masknet3!COCO_datasets!centered_imdb!lr2e-06_wd0_mom0p9_batch30_M224_f300!net-epoch-5pmat', ...
    'deepmask_dag\VOC2012\pascal_imdb\lr1e-06_wd5e-05_mom0p9_batch40_preInitModelPathdata!experiments!deepmask_dag!COCO_datasets!centered_imdb!lr1e-06_wd5e-05_mom0p9_batch40!net-epoch-3pmat'};   

%% Plot IoU

compare_experiments(experimentDirs);
ylabel 'IoU';
xlabel 'No. of epochs';
axis([1 20 -Inf Inf]);

% WORKAROUND - Create four false plots to change legend icons (using the same
% colors used in compare_experiments)
colors = distinguishable_colors(4,[1 1 1]);
for i = 1 : 4
    fakeplot{i} = plot(NaN,'o');
    fakeplot{i}.MarkerFaceColor = colors(i,:); fakeplot{i}.MarkerEdgeColor ='none';
end
legend([fakeplot{1}, fakeplot{2}, fakeplot{3}, fakeplot{4}], 'Merging strategy 3', 'No partial masks', 'Merging strategy 3 (P)', 'No partial masks (P)', ...
       'Location', 'southeast');
   

%% Plot loss

compare_experiments(experimentDirs,'error','objective');
ylabel 'Loss'
xlabel 'No. of epochs'
axis([1 20 -Inf Inf]);

% WORKAROUND - Create four false plots to change legend icons (using the same
% colors used in compare_experiments)
colors = distinguishable_colors(4,[1 1 1]);
for i = 1 : 4
    fakeplot{i} = plot(NaN,'o');
    fakeplot{i}.MarkerFaceColor = colors(i,:); fakeplot{i}.MarkerEdgeColor ='none';
end
legend([fakeplot{1}, fakeplot{2}, fakeplot{3}, fakeplot{4}], 'Merging strategy 3', 'No partial masks', 'Merging strategy 3 (P)', 'No partial masks (P)', ...
       'Location', 'northwest');

