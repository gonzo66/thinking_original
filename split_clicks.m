num_splits=1000;
num_data=length(clicktext);
click_dir='/home/gvaca/ECCV2014/Bing/Train/search/imagedata/';
step=ceil(num_data/1000)
for iSplit=1:num_splits
   start=(iSplit-1)*step+1;
   myend= min((iSplit)*step,num_data);
   click=clicktext(start:myend) ;
   save(sprintf('%s/click%05d.mat',click_dir,iSplit),'click','-v7.3')
end