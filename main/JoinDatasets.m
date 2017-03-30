clear; clc;

% Load training data
load pascal_imdb_train.mat;
nTEx = size(imdb,4);
imdbFull = imdb;
masksFull = masks;
partial_masksFull = partial_masks;
clear imdb masks partial_masks;

% Load validations data
load pascal_imdb_val.mat;
nValEx = size(imdb,4);
imdbFull(:,:,: , nTEx+1 : nTEx+nValEx) = imdb;
masksFull(:,:,: , nTEx+1 : nTEx+nValEx) = masks;
partial_masksFull(:,:,: , nTEx+1 : nTEx+nValEx) = partial_masks;
clear imdb masks partial_masks;

% Create new file and save to it
file = matfile('pascal_imdb.mat');
file.imdb = imdbFull;
file.masks = masksFull;
file.partial_masks = partial_masksFull;

% Create variable to denote which set (train/val) each image belongs to
file.sets = [ones(1,nTEx) 2*ones(1,nValEx)];