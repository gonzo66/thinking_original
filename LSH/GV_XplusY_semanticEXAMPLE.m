% Load the output of 200 NN.
load('LSH/ret_dev_INsemantic.mat');   %nnids2 contains the ids; nndists2 the distances of first 500 search terms of the full dataset
load('BoWT.mat');  %load textual representation of the input vocabulary %BoWT is an array of  78K x 1M
load('FinalDictionary.mat');   %load vocab (The text of each one of the 78K words) & vec (word representation of the word)
load('clicktext.mat');	%11.9 Gb
load('KernelT.mat');	%load kernel=78K x 78K distance matrix in word space

%load('outDEV.mat');	%load visual features of outputs
load('featIN.mat');	%load visual features of quey inputs
load('word_frequency.mat')
K=5


%imvals=[10,12,15,16,17,18,20,21,27,30,32,36,40,41,45,47,51,52,53,54,55,56,57,58,60,61,63,71,72,73,103,105,108,111,113,127,132,141,150,154,157,162,163,180,187,202,204,205,206,223,224,238,239,243,244,286,287,295,305,344,402,422,435,442,471,504,525,528,550,555,586,588,608,621,624,649,668,694,736,740,764,822,826,827,856,858,864,903,914,935,947,992]

pairs=[8 10 97]

imindx=[00119829,00279771,00119829,00599798,00074702,00356603,00317077,00475531,00231207,00754854,00618622,00966088,00131592,00534822,00242396,00173201,00448621,00611780,00138387,00086250,00907449,00905487,00104101,00679262,00168515,00526659,00417290,00614515,00788566,00948247,00714932,00456653,00368036,00518196,00510376,00928390,00651863,00511169,00019729,00609922,00912323,00641342,00776513,00636680,00838720,00971464,00949977,00409531,00616695,00710389,00401151,00986140,00451605,00836834,00020116,00791038,00652007,00922286,00085873,00305994,00581196,00467136,00970637,00860385,00657078,00997630,00571751,00164228,00129633,00877408,00890982,00855411,00083298,00689146,00906046,00109080,00985876,00712885,00167812,00071289,00535772,00810250,00990705,00223877,00550323,00849426,00690579,00622323,00803584,00053299,00203403,00308164,00192279,00422859,00515172,00422566, 00118833];

indi=1:97;
pos_concepts_indx= nnids2(1:K,indi);
num_ret=20
outdir='/home/gvaca/ECCV2014/Bing/Train/search/outs_ind2/';
if(~exist(outdir,'dir'))
	mkdir(outdir);
end
imgs_dir='/home/gvaca/ECCV2014/Bing/Train/search/imgs/';
%for iInd=1:length(indi)
%	outdir2=sprintf('%s/%03d',outdir,indi(iInd));
%	mkdir(outdir2);
%	for iRet=1:num_ret
%		imval=nnids2(iRet,indi(iInd));
%		group=floor(double(imval)/100000);
%		im=imread(sprintf('%s/part%02d/%08d.jpg',imgs_dir,group,imval));
%		imwrite(im,sprintf('%s/N%05d.jpg',outdir2,iRet));
%	end
%
%end
% Let's weight the core concepts
nndists2=exp(-nndists2/mean(nndists2(:)));

iAttempt=0
combina=[]




for in=1:size(pairs,1)
%for in1=1:1:64
% for in2=in1+1:64
	iAttempt=iAttempt+1
	%combina=[combina; in1 in2];

	
	
	% positive index
	pos_ind=pairs(in,:);

	% negative index
	neg_ind=[]


	% let's operate with positive first

	% Get K nearest Neighbors of selected pair
	pos_concepts_indx= nnids2(1:K,pos_ind);
	neg_concepts_indx= nnids2(1:K,neg_ind);
	% Get textual representation
	% Main concept
	core_concepts=BoWT(:,pos_concepts_indx(:,1));
	add_concepts=BoWT(:,pos_concepts_indx(:,2));
	add_concepts2=BoWT(:,pos_concepts_indx(:,3));

%	neg_concepts=BoWT(:,neg_concepts_indx);



	w_pos=double(nndists2(1:K,pos_ind));
	w_neg=double(nndists2(1:K,neg_ind));

	core_concepts=sparse(core_concepts*w_pos(:,1));
	add_concepts=sparse(add_concepts*w_pos(:,2));
	add_concepts2=sparse(add_concepts2*w_pos(:,3));
%	neg_concepts=sparse(neg_concepts*w_neg);

	% Find new concepts
	[vals1, concep1]=sort(core_concepts,'descend');
	[vals2, concep2]=sort(add_concepts,'descend');
	[vals3, concep3]=sort(add_concepts2,'descend');

%	% Compare term by term to create a new concept
	new_candidates=[]
	exclude_list={'pictures' , 'picture', 'the' , 'in', 'for', 'on', 'is', 'with','from','how','free','kids', 'white', 'art', 'image', 'photo',    'photos', 'images', 'clip',  'pics', 'pic',  'wallpaper', 'wallpapers', 'www','printable',  'clipart' ,'about' ,'by', 'that','birthday','whitney'};  % These are the words with frequency over 50.000
	    

do_it2=1;
nnneighs=0;
while do_it2==1
knum=0;
nnneighs=nnneighs+3;
do_it=1;
while do_it==1
knum=knum+50
	%for iC1=1:ceil(length(nonzeros(vals1))/5)
	for iC1=1:nnneighs
		if sum(strcmp(exclude_list,vocab(concep1(iC1))))==0
			[val_concep1,cand_concep1]=sort(kernel(concep1(iC1),:),'descend');
 
		%	for iC2=1:ceil(length(nonzeros(vals2))/8))
			for iC2=1:nnneighs
				if sum(strcmp(exclude_list,vocab(concep2(iC2))))==0
					[val_concep2,cand_concep2]=sort(kernel(concep2(iC2),:),'descend');

			for iC3=1:nnneighs
			        [val_concep3,cand_concep3]=sort(kernel(concep3(iC3),:),'descend');
				[candidate1, ia, ib]=intersect(cand_concep1(1:knum),cand_concep2(1:knum)); 
				[candidate, ia, ib]=intersect(candidate1,cand_concep3(1:knum)); 
					%if length(candidate)>5
					%	candidate=candidate(1:10-max(iC1,iC2));
					%end
					new_candidates=[new_candidates candidate];
			end
				end
			end
		end
	end
	if length(new_candidates)>0
	do_it=0
	do_it2=0
	conc{iAttempt}=vocab(new_candidates)
	end
	if(knum==200)
	    do_it=0;	
	end
end
end

	% Select candidates from concepts
	% Get a list of images that contains the words selected (operate in that subset only)
	candidates=[]
	for iC1=1:10
		if sum(strcmp(exclude_list,vocab(concep1(iC1))))==0
			candidates=[candidates find(BoWT(concep1(iC1),:))];
		end
	end
	for iC2=1:10
		if sum(strcmp(exclude_list,vocab(concep2(iC2))))==0
			candidates=[candidates find(BoWT(concep2(iC2),:))];
		end
	end
	for iC3=1:10
		if sum(strcmp(exclude_list,vocab(concep3(iC3))))==0
			candidates=[candidates find(BoWT(concep3(iC3),:))];
		end
	end
	for iC3=1:length(new_candidates)
%		if length(new_candidates)>=iC3
			if sum(strcmp(exclude_list,vocab(new_candidates(iC3))))==0
				candidates=[candidates find(BoWT(new_candidates(iC3),:))];
			end
%		end
	end
	candidates=unique(candidates);

	% Sum up all the concepts in the histogram
	textrep= 0.1*core_concepts+ 0.1*sum(add_concepts,2)+ 0.1*sum(add_concepts2,2);
	for iCand=1:length(new_candidates)
		textrep(new_candidates(iCand))=textrep(new_candidates(iCand))+max(core_concepts)*(min(3,length(new_candidates)))/(sqrt(length(new_candidates)));
	end

	%textrep=zeros(size(core_concepts,1),size(core_concepts,2));
	%textrep(new_candidates)=1;

	%Now start doing td_idf on candidates only
	num_concepts=nnz(textrep);
	concept_vals=nonzeros(textrep);
	concept_ids=find(textrep);
	scores=zeros(1,length(candidates));	
	for iHist=1:num_concepts
		% add the score of each concept of all the candidate images
		%scores=scores+concept_vals(iHist)*BoWT(concept_ids(iHist),candidates);
		scores=scores+concept_vals(iHist)*BoWT(concept_ids(iHist),candidates)./(freq(concept_ids(iHist)));
	end
	[scoreval,ind]=sort(scores,'descend');

	%save some output images
	num_ret=150;
	num_list=500;
	retrieved_list=candidates(ind(1:num_list));
	outdir='/home/gvaca/ECCV2014/Bing/Train/search/outs_semanticEXAMPLE/';
	imgs_dir='/home/gvaca/ECCV2014/Bing/Train/search/imgs/';
	if(~exist(outdir,'dir'))
		mkdir(outdir);
	end
	outdir2=sprintf('%s/%03d',outdir,iAttempt);
	mkdir(outdir2);
	for iRet=1:num_ret
		imval=retrieved_list(iRet);
		group=floor(imval/100000);
		im=imread(sprintf('%s/part%02d/%08d.jpg',imgs_dir,group,imval));
		imwrite(im,sprintf('%s/N%05d.jpg',outdir2,iRet));
	end

	% save input images
	querydir='/home/gvaca/ECCV2014/Bing/Dev/imgs/';
	querydir='/home/gvaca/ECCV2014/Bing/Train/search/imgs/';

	for iInput=1:length(pos_ind)
		%imtemp=imread(sprintf('%s/img%d.jpg',querydir,pos_ind(iInput)-1));
queryval=imindx(pos_ind(iInput));
mygroup=floor(double(queryval)/100000);
imtemp=imread(sprintf('%s/part%02d/%08d.jpg',querydir,mygroup,queryval));
		imwrite(imtemp,sprintf('%s/I%02d.jpg',outdir2,iInput));
	end

	% Visual Re-ranking
	

	%find distances of the query images to retrieved images and pick the most similar, then weight the score based on this distance
%	query_des=feat_imageNet(pos_ind,:);
	query_des=featIN(imindx(pos_ind),:);
	ret_desc= featIN(retrieved_list,:);
	distances=zeros(size(ret_desc,1),size(query_des,1));

	for i=1:length(pos_ind)
		tmp=(ret_desc-repmat(query_des(i,:),size(ret_desc,1),1)).^2 ;
		distances(:,i)=exp(-sum(tmp,2)/10000);
	end
	distances=max(distances,[],2);

	scoreval2=scoreval(1:num_list).*distances';
	[vals,rerank]=sort(scoreval2(1:num_list),'descend');
	rerank=retrieved_list(rerank);
	for iRet=1:num_ret
		imval=rerank(iRet);
		group=floor(imval/100000);
		im=imread(sprintf('%s/part%02d/%08d.jpg',imgs_dir,group,imval));
		imwrite(im,sprintf('%s/R%05d.jpg',outdir2,iRet));
	end

%end
end
