%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% README FILE.
% Make a LSH hashing of the training data for all the available features
% Gonzalo Vaca-Castano. CRCV 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the training data with all the features
% train_featALL contains all the features
%load('../train_featTOTAL.mat');
%load('../outHolidays_1k.mat')
load('../outHolidays.mat')
%load('../outFlick100K.mat')
%feat_imageNet=[feat_100K ; feat_imageNet];
feat_imageNet=feat_imageNet';

% Load the testing data
%load('../test_featALL.mat');
load('../querysHolidays.mat')


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
%ret_outs=cell(20,1);

feat_Query=feat_imageNet(:,Qindex);


		
	dim=size(feat_imageNet,1);
	num_images=size(feat_imageNet,2);   %num_features
	%index = ccvLshCreate(ntables, nfuncs, 'l2', 'l2', 1, dim, .1,1000);
	index = ccvLshCreate(ntables, nfuncs, 'l1', 'kl', 1, dim, .1, 1000);
	ccvLshInsert(index, feat_imageNet);
%	ccvLshSave(index, sprintf('lsh_dev_IN_Oxford.mat'));
%	index= ccvLshLoad(sprintf('lsh_dev_IN_Oxford.mat'));

%	num_neig=1000;
	num_neig=num_images
	num_queries=length(Qindex);

	[nnids2 nndists2] = ccvLshKnn(index, feat_imageNet , feat_Query(:,1:num_queries), num_neig);
	impath='/home/gvaca/ECCV2014/Bing/Train/search/imgs/';
%	ret_outs.nnids=nnids2;
%	ret_outs.nndists=nndists2;

%	for iQuery=1:num_queries
%	    neigs=nnids2(:,iQuery);
%	    neigs=neigs(neigs~=0);
%	    retrieved=length(neigs);
%	    dim1=floor(sqrt(retrieved));
%	    dim2=ceil(retrieved/dim1);
%	    mkdir(sprintf('outs2/%03d/',iQuery));
	 
%	    for iRetrieval=1:retrieved
%
%		img_ret =neigs(iRetrieval)
%		part=floor(double(img_ret)/100000);
%		img_retfile=sprintf('%s/part%02d/%08d.jpg',impath,part,img_ret)
%		im=imread(img_retfile);
%		imwrite(im,sprintf('outs2/%03d/IN-%04d.png',iQuery,iRetrieval));
%	    end	
%	end
%	ccvLshClean(index);

save('ret_dev_IN_Holidays.mat','nnids2', 'nndists2','-v7.3');  
