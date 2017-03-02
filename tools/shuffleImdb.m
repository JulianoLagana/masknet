% imdb file must be inside a 'data' folder
imdb_name = 'partial_imdb.mat';

file = matfile(['data/' imdb_name]);
sizes = getfield(whos(file),'size');
w = sizes(1);
h = sizes(2);
c = sizes(3);
nImages = sizes(4);
newIdxs = randperm(nImages);

new_name = ['data/shuffled_' imdb_name];
new_file = matfile(new_name);

% Create the first two dummy entries, so matlab knows its a 4D variable
new_file.imdb(1:w,1:h,1:c,1:2) = uint8(0);
new_file.masks(1:w,1:h,1,1:2) = uint8(0);

% Copy each file in a random order
progressTick = round(nImages/100);
handleWaitBar = waitbar(0,'Please wait.');
for i = 1 : nImages
    new_file.imdb(1:w,1:h,1:c,i) = file.imdb(:,:,:,newIdxs(i));
    new_file.masks(1:w,1:h,1,i) = file.masks(:,:,:,newIdxs(i));
    
    % If it's the right time, update the progress bar
    if mod(i,progressTick) == 0
        progress = i/nImages;
        msg = sprintf('Please wait: %i%% complete',round(progress*100));
        waitbar(progress,handleWaitBar, msg);
    end
end
close(handleWaitBar);

