addpath segmenter;
addpath main;
addpath(genpath('masknet'));
addpath(genpath('detector'));
    rmpath('detector/SelectiveSearchCodeIJCV/Dependencies/anigaussm');
    rmpath('detector/SelectiveSearchCodeIJCV/Dependencies/FelzenSegment');
addpath(genpath('PascalVocdevkit'));
addpath(genpath('report'));