% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to load train clicks file of the Bing dataset. 
% queries get indexed. Asociated images are showed 
%
% Gonzalo Vaca-Castano
% CRCV. 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


tsvfile='TrainClickLog.tsv';            %Click files

%load('nametest.mat')                    % name of the files
count=0;

fid2=fopen(tsvfile,'r');
tline = fgetl(fid2);

%clicktext=cell(1000000,1);
prevname='';
mapWord = containers.Map;

count2=0;
tic
while ischar(tline)
    count=count+1;
    [name rest] = strtok(tline, char(9));
    [text clicks] = strtok(rest, char(9));
    clicks=str2num(clicks);
    %if(~strcmp(name,prevname))
        index=find(strcmp(name,nametest));                
    %end
    
    tf = isKey(mapWord,text)
    if tf == 0
        count2=count2+1;
        %num_elem=0;
        %clicktext{index}.clicks= [clicks];
        mapWord(text)=index;
        if(mod(count2,1000)==0)
           fprintf('Processed: %f %% \r',count2/10000); 
           toc
        end
    else
        mapWord(text)=[mapWord(text) ; index];         
    end
    
    tline = fgetl(fid2);
end
save('mapWord.mat','mapWord','-v7.3')
fclose(fid2);