fid=fopen('conceplist_part1.txt','w')
for i=1:120
    fprintf(fid,'#### folder %03d #####\n',i);
    fprintf(fid,'%s \n',newconc{i}{:});    
    fprintf(fid,'\n');
end
fclose(fid)