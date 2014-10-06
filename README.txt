%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% README FILE.
% Search process of the retrieval system
% Gonzalo Vaca-Castano. CRCV 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

* FILES & DIRECTORIES STRUCTURE
- train_featTOTAL2.mat: Contains all the features for the training data including faces information
- mapWordFinal.mat: contains the hash table for all the text in the training table. each key has a list of all files that containst the key
- clicktext.mat: contains a description of the corresponding tags associated to each image
- imgs/ : directory with all the training images grouped in 100,000 images per folder
- /home/gvaca/ECCV2014/Bing/Train/faces/ : directory with all the faces found in the training dataset
- featIN.mat: features of Bing's dataset training data using ImageNet method
- outDEV.mat: features of Bing's dataset testing data using ImageNet method
- FinalDictionary.mat: contains the vector representation of the words  that are present in our dataset
- BoWT.mat: Contains the textual BoW representatioon of 1 Million images of the dataset
- word_frequency.mat: frequencies of each word. ised in tf-idf
