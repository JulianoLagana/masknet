VOCinit;

% Choose the part of the dataset to display
imgset = 'train';

% Read all image ids in the chosen Pascal VOC set
ids = textread(sprintf(VOCopts.seg.imgsetpath,imgset),'%s');

% For each image
for i = 1 : numel(ids)
    % Get the paths to the images, segmentations and annotations
    imgpath = sprintf(VOCopts.imgpath,ids{i});
    annopath = sprintf(VOCopts.annopath,ids{i});
    clssegpath = sprintf(VOCopts.seg.clsimgpath,ids{i});
    objsegpath = sprintf(VOCopts.seg.instimgpath,ids{i});
    
    % Load the images, segmentations and annotations
    ann = PASreadrecord(annopath);
    img = imread(imgpath);
    clsseg = imread(clssegpath);
    objseg = imread(objsegpath);
    
    % Segment the image using FCN-8s
    net = dagnn.DagNN.loadobj(load('data/models/pascal-fcn8s-dag.mat')) ;
    segFCN = segment(net, img);
    clear net;
    sz = size(img);
    sz = sz(1:2);
    segFCN = imresize(segFCN,sz,'nearest');
    
    % Generate pre-proposals using selective search
    proposals = generateProposals(img);
    
    % Generate bounding boxes using fast-rcnn
    bboxes = run_fast_rcnn(img,proposals);
    
    subplot(2,2,1);
    imshow(img); hold on;
    draw_bboxes_test(bboxes);
    hold off;
    subplot(2,2,2);
    cMap = VOClabelcolormap(256);
    imshow(clsseg,cMap);
    subplot(2,2,3);
    imshow(segFCN,cMap);
    
    waitforbuttonpress;
    
end

function draw_bboxes_test(bboxes)

    cat_names = {'background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat', ...
    'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant', ...
    'sheep','sofa','train','tvmonitor'};

    bboxes = double(bboxes);

    for i = 1 : size(bboxes,1)
        rectangle('Position',bboxes(i,1:4),'LineWidth',3,'EdgeColor','r');
        text(bboxes(i,1), bboxes(i,2)-5, cat_names{bboxes(i,end)}, 'FontSize', 10);
    end
    

end