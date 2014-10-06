%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% README FILE.
% Make a LSH hashing of the training data for all the available features
% Gonzalo Vaca-Castano. CRCV 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the training data with all the features
% train_featALL contains all the features
load('../train_featTOTAL.mat');
dim=size(train_featALL.Gist,1);
num_images=size(train_featALL.Gist,2);   %num_features


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
index = ccvLshCreate(ntables, nfuncs, 'l2', 'l2', 1, dim, .1, 1000);
ccvLshInsert(index, train_featALL.Gist);

[nnids nndists] = ccvLshKnn(index, train_featALL.Gist , train_featALL.Gist(:,1:10), 2);
ccvLshSave(index, 'lsh_gist.mat');

index= ccvLshLoad('lsh_gist.mat');

num_neig=10;
num_queries=10;
[nnids2 nndists2] = ccvLshKnn(index, train_featALL.Gist , train_featALL.Gist(:,1:num_queries), num_neig);
impath='/home/gvaca/ECCV2014/Bing/Train/search/imgs/';

for iQuery=1:num_queries
    neigs=nnids2(:,iQuery);
    neigs=neigs(neigs~=0);
    retrieved=length(neigs);
    dim1=floor(sqrt(retrieved));
    dim2=ceil(retrieved/dim1);
    mkdir(sprintf('outs/%03d/',iQuery));
 %   h1=figure(1);
    for iRetrieval=1:retrieved
%	subplot(dim1,dim2,iRetrieval);
	img_ret =neigs(iRetrieval)
	part=floor(double(img_ret)/100000);
	img_retfile=sprintf('%s/part%02d/%08d.jpg',impath,part,img_ret)
	im=imread(img_retfile);
%	imshow(im);
	imwrite(im,sprintf('outs/%03d/%04d.png',iQuery,iRetrieval));
    end
	
%     print(h1,'-dpng',sprintf('outs/fig%03d.png',iQuery));
end


ccvLshClean(index);
      
