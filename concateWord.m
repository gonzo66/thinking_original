%load first value

load('BoW00001.mat');
BoWT=sparse(boWImgs);


for i=2:1000
    clear('boWImgs');
    load(sprintf('BoW%05d.mat',i));
    BoWT=[BoWT sparse(boWImgs)];
end
