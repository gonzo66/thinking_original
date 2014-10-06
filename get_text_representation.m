% find the reduced vocabulary and image representation
function get_text_representation(iBatch)
load('GoogleNews-vectors-negative300.bin');
keydir='/home/ashan/gonzalo/ECCV2014/BingChallenge/keys/';
load(sprintf('%s/key%05d.mat',keydir,iBatch ));

%rep=cell(length(vocabulary),1);
words=[];
for iWord=1:length(vocabulary)
    stringval=keys(iWord);
    remain = stringval;
    %words=[];
    while true
     [str, remain] = strtok(remain, ' ');
     if isempty(str),  break;  end
        %disp(sprintf('%s', str))
        words=[words find(strcmp(vocabulary,str))]                
    end
end
save(sprintf('%s/index%05d.mat',keydir,iBatch))
end