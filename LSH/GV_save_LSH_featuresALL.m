%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% README FILE.
% Make a LSH hashing of the training data for all the available features
% Gonzalo Vaca-Castano. CRCV 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the training data with all the features
% train_featALL contains all the features
load('../train_featTOTAL.mat');

% Load the testing data
load('../test_featALL.mat');



% Let's start with gist
% INPUTS
% ------
% ntables   - [1] the number of tables
% nfuncs    - [5] the number of hash functions per table
% htype     - ['l2'] the type of hash function to use
%             'ham' -> use hamming distance, h = x[i]
%             'l1'  -> use l1 distance, h = floor((x[i]-b) / w)
%             'l2'  -> use l2 distance, h = floor((x . r - b) / w)
%             'cos' -> use cons ditance, h = sign(x . r)
%             'min-hash'  -> use min-hash function, h = min(perm(x))
%             'sph-sim'   -> use spherical simplex function
%             'sph-orth'  -> use spherical orthoplex
%             'sph-hyp'   -> use shperical hypercube
%             'bin-gauss' -> binary gaussian kernels
% dist      - ['l2'] the distance type to use
%             'l1'      -> l1 distance
%             'l2'      -> l2
%             'hamming' -> hamming
%             'cos'     -> dot-product
%             'arcos'   -> acos of dot-product
%             'bhat'    -> Bhattacharya distance
%             'kl'      -> KL-divergence
%             'jac'     -> Jacquard distance
%             'xor'     -> XOR distance for packed binary numbers
% norm      - [1] normalize or not
% ndims     - the number of dimensions for the input data
% w         - [.25] the size of the bin for 'l2' and 'l1' hash functions
% tsize     - [1000] the size of the table if it's a fixed size table, 
%             or 0 for a variable sized hash table
% seed      - random seed for input
% hwidth    - [0] number of bits of every hash value. In case hwidth~=0,
%             the outputs of the different hash functions are OR'ed
%             together at the right place, with function 1 at place
%             nfuncs*hwidth and function N at place 0 (LSB).
% bitsperdim - [0] number of bits per input dimension. If nonzero, every
%             dimension represents bitsperdim binary bits, in which case,
%             the number of dimensions for Hamming hash function is
%             bitsperdim*ndims. This is useful only for 'ham' hash function.
%


type='lsh-l2'
ntables = 4;
nfuncs = 10;
ret_outs=cell(20,1);

for iFeat=1:20
	switch iFeat
          case 1
		feature=train_featALL.Gist;
		featuretest=test_featALL.Gist;
          case 2
		feature=train_featALL.colorHisHSV;
		featuretest=test_featALL.colorHisHSV;
	  case 3
		feature=train_featALL.colorHisRGB;
		featuretest=test_featALL.colorHisRGB;		
	  case 4
		feature=train_featALL.colorHisLab;
		featuretest=test_featALL.colorHisLab;
	  case 5
		feature=train_featALL.momentsHSV;
		featuretest=test_featALL.momentsHSV;
	  case 6
		feature=train_featALL.momentsRGB;
		featuretest=test_featALL.momentsRGB;
	  case 7
		feature=train_featALL.momentsLab;
		featuretest=test_featALL.momentsLab;
	  case 8
		feature=train_featALL.SPcolorHisHSV;
		featuretest=test_featALL.SPcolorHisHSV;
	  case 9
		feature=train_featALL.SPcolorHisRGB;
		featuretest=test_featALL.SPcolorHisRGB;
	  case 10
		feature=train_featALL.SPcolorHisLab;
		featuretest=test_featALL.SPcolorHisLab;
	  case 11
		feature=train_featALL.SPmomentsHSV;
		featuretest=test_featALL.SPmomentsHSV;
	  case 12
		feature=train_featALL.SPmomentsRGB;
		featuretest=test_featALL.SPmomentsRGB;
	  case 13
		feature=train_featALL.SPmomentsLab;
		featuretest=test_featALL.SPmomentsLab;
	  case 14
		feature=train_featALL.thumbnails;
		featuretest=test_featALL.thumbnails;
	  case 15
		feature=train_featALL.HOGfull;
		featuretest=test_featALL.HOGfull;
	  case 16
		feature=train_featALL.num_faces;
		featuretest=test_featALL.num_faces;
		continue;
	  case 17
		feature=train_featALL.LBPface;		
		featuretest=test_featALL.LBPface;
	  case 18
		feature=train_featALL.LBPface2;
		featuretest=test_featALL.LBPface2;
	  case 19
		feature=train_featALL.HOGface;
		featuretest=test_featALL.HOGface;
 	  case 20
		feature=train_featALL.HOGface2;
		featuretest=test_featALL.HOGface2;
	end

		
	dim=size(feature,1);
	num_images=size(feature,2);   %num_features
	index = ccvLshCreate(ntables, nfuncs, 'l2', 'l2', 1, dim, .1, 1000);
	ccvLshIlgnsert(index, feature);
	ccvLshSave(index, sprintf('lsh_%03d.mat',iFeat));

	index= ccvLshLoad(sprintf('lsh_%03d.mat',iFeat));

	num_neig=20;
	num_queries=20;
	[nnids2 nndists2] = ccvLshKnn(index, feature , featuretest(:,1:num_queries), num_neig);
	impath='/home/gvaca/ECCV2014/Bing/Train/search/imgs/';
	ret_outs{iFeat}.nnids=nnids2;
	ret_outs{iFeat}.nndists=nndists2;

	for iQuery=1:num_queries
	    neigs=nnids2(:,iQuery);
	    neigs=neigs(neigs~=0);
	    retrieved=length(neigs);
	    dim1=floor(sqrt(retrieved));
	    dim2=ceil(retrieved/dim1);
	    mkdir(sprintf('outs2/%03d/',iQuery));
	 
	    for iRetrieval=1:retrieved

		img_ret =neigs(iRetrieval)
		part=floor(double(img_ret)/100000);
		img_retfile=sprintf('%s/part%02d/%08d.jpg',impath,part,img_ret)
		im=imread(img_retfile);
		imwrite(im,sprintf('outs/%03d/%03d-%04d.png',iQuery,iFeat,iRetrieval));
	    end	
	end
	ccvLshClean(index);
end
save('ret_outs_dev.mat','ret_outs','-v7.3');  
