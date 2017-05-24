experimentDirs = ...
    {'masknet3\VOC2012\pascal_imdb\lr2e-06_wd0_mom0p9_batch30_M224_f300', ...
    'deepmask_dag\VOC2012\pascal_imdb\lr1e-06_wd5e-05_mom0p9_batch40', ...
    'masknet3\VOC2012\pascal_imdb\lr1e-06_wd0_mom0p9_batch30_preInitModelPathdata!experiments!masknet3!COCO_datasets!centered_imdb!lr2e-06_wd0_mom0p9_batch30_M224_f300!net-epoch-5pmat', ...
    'deepmask_dag\VOC2012\pascal_imdb\lr1e-06_wd5e-05_mom0p9_batch40_preInitModelPathdata!experiments!deepmask_dag!COCO_datasets!centered_imdb!lr1e-06_wd5e-05_mom0p9_batch40!net-epoch-3pmat'};   

compare_experiments(experimentDirs);
legend 'Merging strategy 3' 'No partial masks' 'Merging strategy 3 (P)' 'No partial masks (P)';
ylabel 'IoU';
xlabel 'No. of epochs';

compare_experiments(experimentDirs,'error','objective');
legend 'Merging strategy 3' 'No partial masks' 'Merging strategy 3 (P)' 'No partial masks (P)';
ylabel 'Loss'
xlabel 'No. of epochs'

