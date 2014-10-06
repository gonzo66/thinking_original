%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% README FILE.
% Make a LSH hashing of the training data for all the available features
% Gonzalo Vaca-Castano. CRCV 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the training data with all the features
% train_featALL contains all the features
%load('../train_featTOTAL.mat');
load('../featIN.mat')

% Load the testing data
%load('../test_featALL.mat');
load('../outDEV.mat')


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
ntables = 40;
nfuncs = 100;
%ret_outs=cell(20,1);

featIN=featIN';
feat_imageNet=feat_imageNet';

		
	dim=size(featIN,1);
	num_images=size(featIN,2);   %num_features
	index = ccvLshCreate(ntables, nfuncs, 'l1', 'l2', 1, dim, .1, 1000);
	ccvLshInsert(index, featIN);
	ccvLshSave(index, sprintf('lsh_dev_INsemantic.mat'));
imgIndx=[00119829 00279771 00119829 00599798 00074702 00356603 00317077 00475531 00231207 00754854 00618622 00966088 00131592 00534822 00242396 00173201 00448621 00611780 00138387 00086250 00907449 00905487 00104101 00679262 00168515 00526659 00417290 00614515 00788566 00948247 00714932 00456653 00368036 00518196 00510376 00928390 00651863 00511169 00019729 00609922 00912323 00641342 00776513 00636680 00838720 00971464 00949977 00409531 00616695 00710389 00401151 00986140 00451605 00836834 00020116 00791038 00652007 00922286 00085873 00305994 00581196 00467136 00970637 00860385 00657078 00997630 00571751 00164228 00129633 00877408 00890982 00855411 00083298 00689146 00906046 00109080 00985876 00712885 00167812 00071289 00535772 00810250 00990705 00223877 00550323 00849426 00690579 00622323 00803584 00053299 00203403 00308164 00192279 00422859 00515172 00422566 00118833];

%	index= ccvLshLoad(sprintf('lsh_dev_IN2.mat'));

	num_neig=200;
	num_queries=20;


%	[nnids2 nndists2] = ccvLshKnn(index, featIN , feat_imageNet(:,1:num_queries), num_neig);
	[nnids2 nndists2] = ccvLshKnn(index, featIN , featIN(:,imgIndx), num_neig);
	save('ret_dev_INsemantic.mat','nnids2', 'nndists2','-v7.3');  
	
	impath='/home/gvaca/ECCV2014/Bing/Train/search/imgs/';
%	ret_outs.nnids=nnids2;
%	ret_outs.nndists=nndists2;
num_queries=97
	for iQuery=1:num_queries
	    neigs=nnids2(:,iQuery);
	    neigs=neigs(neigs~=0);
	    %retrieved=length(neigs);
retrieved=min(20,length(neigs))
	    dim1=floor(sqrt(retrieved));
	    dim2=ceil(retrieved/dim1);
	    mkdir(sprintf('outs_semantic/%03d/',iQuery));
	 
	    for iRetrieval=1:retrieved

		img_ret =neigs(iRetrieval)
		part=floor(double(img_ret)/100000);
		img_retfile=sprintf('%s/part%02d/%08d.jpg',impath,part,img_ret)
		im=imread(img_retfile);
		imwrite(im,sprintf('outs_semantic/%03d/IN-%04d.png',iQuery,iRetrieval));
	    end	
	end
	ccvLshClean(index);

%save('ret_dev_INsemantic.mat','nnids2', 'nndists2','-v7.3');  
