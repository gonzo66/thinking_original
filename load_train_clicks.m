% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to load train clicks file of the Bing dataset
%
% Gonzalo Vaca-Castano
% CRCV. 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


tsvfile='TrainClickLog.tsv';            %Click files

load('nametest.mat')                    % name of the files
count=0;

fid2=fopen(tsvfile,'r');
tline = fgetl(fid2);
clicktext=cell(1000000,1);
prevname='';

count2=0;
tic
while ischar(tline)
    count=count+1;
    [name rest] = strtok(tline, char(9));
    [text clicks] = strtok(rest, char(9));
    clicks=str2num(clicks);
    if(~strcmp(name,prevname))
        index=find(strcmp(name,nametest));                
    end
    
    if isempty(clicktext{index})
        count2=count2+1;
        num_elem=0;
        clicktext{index}.clicks= [clicks];
        if(mod(count2,1000)==0)
           fprintf('Processed: %f %% \r',count2/10000); 
           toc
        end
    else
         num_elem=length(clicktext{index}.clicks);
         clicktext{index}.clicks= [clicktext{index}.clicks ; clicks];
    end
    clicktext{index}.texts{num_elem+1}= text;
    
    prevname=name;
    tline = fgetl(fid2);
end
save('clicktext.mat','clicktext','-v7.3')
fclose(fid2);
