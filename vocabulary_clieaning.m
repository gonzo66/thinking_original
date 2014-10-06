% Script to remove noisy words from vocabulary

%load('mapWordFinal.mat')
%load('clicktext.mat')
%allKeys = keys(mapWord);

%split the keys
num_splits=1000;
keys_dir='/home/ashan/gonzalo/ECCV2014/BingChallenge/keys/';
step=ceil(mapWord.Count/1000)
for iSplit=1:num_splits
   start=(iSplit-1)*step+1;
   myend= min((iSplit)*step,mapWord.Count);
   keys=allKeys(start:myend) ;
   save(sprintf('%s/key%05d.mat',keys_dir,iSplit),'keys','-v7.3')
end

