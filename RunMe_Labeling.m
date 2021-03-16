%%Journal Code -------------------------------------------------
% This code tests the effect of using different texture descriptors
% for seismic scene labeling.
% Code obtained from Yazeed
% Last edited by Motaz on 1/26/2017
%------------------------------------------------------------------------
% --------------------------------------------------------------------- %
%                       Clear, load data, and prepare                   %
% --------------------------------------------------------------------- %
clc; 
clear all; 
close all; 


s = rng(1); 
addpath(genpath('./Support functions')); 
addpath(genpath('./LabelingCodes')); 

%% Loading data 

DataCH = load('./LabelingCodes/Data/Patches_ch.mat');
DataSM = load('./LabelingCodes/Data/Patches_sm.mat');
DataHO = load('./LabelingCodes/Data/Patches_ho.mat');
DataFA = load('./LabelingCodes/Data/Patches_fa.mat');
DataS1 = load('./LabelingCodes/Data/Patches_sa1.mat');
DataS2 = load('./LabelingCodes/Data/Patches_sa2.mat');
DataS3 = load('./LabelingCodes/Data/Patches_sa3.mat');

DataCH = DataCH.Patches_ch; % chaotic
DataSM = DataSM.Patches_sm; % smooth
DataHO = DataHO.Patches_ho; % horizon
DataOT = cat(1, DataSM, DataHO); %other class
DataFA = DataFA.Patches_fa; % faults
DataS1 = DataS1.Patches_sa1;
DataS2 = DataS2.Patches_sa2;
DataS3 = DataS3.Patches_sa3;
DataSA = cat(1, DataS1, DataS2, DataS3); % salt domes

% number of images in each class:
chx = size(DataCH,1);
otx = size(DataOT,1);
fax = size(DataFA,1);
sax = size(DataSA,1);

numImages = chx + otx + fax + sax;

% make the images into sets of horizontal vectors
VectorsCH = reshape(DataCH, chx, 99*99);
VectorsOT = reshape(DataOT, otx, 99*99);
VectorsFA = reshape(DataFA, fax, 99*99);
VectorsSA = reshape(DataSA, sax, 99*99);


% combine data into one single matrix
x = [VectorsCH; VectorsOT; VectorsFA; VectorsSA];

% Construct labels vector
y = cell(numImages,1);
for i = 1:numImages
if i <= chx
y{i} = 'Chaotic';
elseif i<= chx+otx
y{i} = 'Other';
elseif i<= chx+otx+fax
y{i} = 'Fault';
elseif i<= chx+otx+fax+sax
y{i} = 'Salt';
end
end
y = y';

% randomly shift the images (rows)
rng(1);
rndIndices = randperm(numImages);
X = x(rndIndices,:);
Y = y(rndIndices);



load('crossline.mat');

% load GT:

load('./LabelingCodes/Data/GT61.mat')
load('./LabelingCodes/Data/GT211.mat')
load('./LabelingCodes/Data/GT231.mat')
load('./LabelingCodes/Data/GT281.mat')


%% --------------------------------------------------------------------- %
%                         Apply Attributes to Data                       %
% ---------------------------------------------------------------------- %
%option = [{'Amplitude'},{'GaussianPyr'},{'Gabor'},{'DWT'},{'SWT'},{'SP'},{'CnT'},{'NCnT'},{'CT'}]; 
option = [{'Amplitude'},{'GaussianPyr'},{'DWT'},{'Gabor'},{'CT'}]; 
%option = {'CT'}
clc;
numDescriptors = length(option); % number of descriptors to test


% first index: 4 = 4 labeled sections
resultsPA = zeros(4,numDescriptors);
resultsMA = zeros(4,numDescriptors);
resultsMIU = zeros(4,numDescriptors);
resultsFWIU = zeros(4,numDescriptors);

%%%%% DEFAULT PARAMETERS %%%%%%
% --------- Main Parameters ----------
% -------------- SVM -----------------
outlierFraction = 0.018; % SVM slack 
numClasses = 4;
costMatrix =  1 - eye(numClasses,numClasses);
CodingVar = 'onevsall';
% ------------ Defaults --------------
windowSize = 99;
sigma = 25; % Sigma for the Gaussian kernel
% --------------- SLIC  --------------
regionSize = 15; % used to be 35
regularizer = 0.3; % or 0.3 or 0.05
calculatePosteriors = 0;

% Make Gaussian Mask:
[Rows,Cols] = ndgrid(1:windowSize, 1:windowSize);
center = (windowSize)/2;
exponent = ((Rows-center).^2 + (Cols-center).^2)./(2*sigma^2);
gMask   = 1*(exp(-exponent));
% imagesc(gMask) surf(gMask)
%%
for descriptor = 1:numDescriptors
    rng(s)
    XX = zeros(numImages, length(GenerateFeatures(reshape(squeeze(X(1,:)),[99 99]),option{descriptor})));
    parfor i = 1:numImages
    old_img = reshape(squeeze(X(i,:)),[99 99]);
    masked_img = gMask.*old_img;
    new_img = GenerateFeatures(masked_img, option{descriptor});
    XX(i,:) = new_img;
    end

       
% ---------------------------------------------------------------------  %
%                           Setup SVM and Train                          %
% ---------------------------------------------------------------------- %
rng(1);
t=templateSVM('CacheSize','maximal','OutlierFraction', outlierFraction, ...
'Standardize',true, 'KernelFunction','linear','Verbose',0, ...
'prior', 'uniform');
Mdl = fitcecoc(XX,Y, 'Learners',t,'Options',statset('UseParallel',1),...
'FitPosterior',1 ,...
'ClassNames',{'Chaotic', 'Other', 'Fault', 'Salt'},...
'prior', 'uniform', 'Coding', 'binarycomplete');

% compute the in-sample error:
isLoss = resubLoss(Mdl);
disp(['Training error is: %' num2str(100*isLoss)]);

% --------------------------------------------------------------------- %
%                            Cross Validation                            %
% ---------------------------------------------------------------------- %
%     crossValidateResults = 0;
%     if crossValidateResults == 1
%         CVMdl = crossval(Mdl,'Kfold' ,5);
%         % estimate Generalization loss (out of sample error):
%         oosLoss = kfoldLoss(CVMdl);
%         disp(['Validation error is: %' num2str(100*oosLoss)]);
%         %     ScoreSVMModel = fitSVMPosterior(CVMdl);
%     end
%
% --------------------------------------------------------------------- %
%                      Label Seismic Section (2D)                        %
% ---------------------------------------------------------------------- %

crosslineNumber = [61,211,231,281];

    for crossline_idx = 1:4
        img = squeeze(Crossline(crosslineNumber(crossline_idx),:,:));
        img = img(105:end, 1:end); % crop TOP 105 PIXELS (noise)
        buffer = (windowSize-1)/2;
        img_ext = padarray(img,[0,buffer],'symmetric', 'both');
        img_ext = padarray(img_ext,[buffer,0],'replicate','both');
        [imHeight, imWidth] = size(img_ext);

        classifiedImage =  uint8(zeros(size(img)));
        confidenceMap =  zeros(size(img));
        posteriorMap = zeros([size(img),numClasses]);

        [gx, gy] = vl_grad(img);
        img3D = cat(3, img, gx, gy);
        img_slic = vl_slic(single(img3D), regionSize, regularizer);
        numSeg = single(max(img_slic(:)));

        for i = 0:numSeg
            region = zeros(size(img_slic));
            xy2D = find(img_slic == i);
            if isempty(xy2D)
                continue;
            end

            region(xy2D) = 1;
            regionProperties = regionprops(region);
            centroid = round(regionProperties.Centroid);
            centroid = centroid(2:-1:1);

            Window = img_ext(centroid(1):centroid(1)+2*buffer, centroid(2):centroid(2)+2*buffer);

            % extract features and evaluate:
            Window = gMask.*Window;


            windowDV = GenerateFeatures(Window, option{descriptor});

            [x,y] = ind2sub(size(img_slic),xy2D);

            if calculatePosteriors == 1
                [label,NegLoss,PBScore,posterior] = predict(Mdl,windowDV);
                confidenceMap(xy2D) = max(posterior)/sum(posterior);
            else
                [label] = predict(Mdl,windowDV);
            end

            if  strcmp(label,'Chaotic')
                classifiedImage(xy2D)  = 2;
            elseif strcmp(label,'Other')
                classifiedImage(xy2D)  = 1; % these numbers correspond to those in ground truth. DOn't change
            elseif strcmp(label,'Fault')
                classifiedImage(xy2D)  = 3;
            elseif strcmp(label,'Salt')
                classifiedImage(xy2D)  = 4;
            end
        end

        % colorize:
        Transperency =  uint8(zeros([size(img),3]));
        [height, width] = size(classifiedImage);
        for i = 1:height
            for j = 1:width
                if classifiedImage(i,j) == 2 % CHaotic
                    Transperency(i,j,:) = [0,0,255]; % blue
                elseif classifiedImage(i,j) == 1 % other:
                    Transperency(i,j,:) = [184,184,184]; % Gray
                elseif classifiedImage(i,j) == 3  % Fault
                    Transperency(i,j,:) = [0,255,0]; % Green
                elseif classifiedImage(i,j) == 4 % salt
                    Transperency(i,j,:) = [255,0,0]; % red
                end
            end
        end

        name = strcat('.\LabelingResults\Crossline',num2str(crosslineNumber(crossline_idx)),'-Descriptor', option{descriptor}, '.bmp');
        a = 0.45; 
        coloredImage = (a*im2double(Transperency))+((1-a)*img); 
        imwrite(coloredImage, name)
        
       
        switch crossline_idx
            case 1
            groundTruth = GT61;
            case 2
            groundTruth = GT211;
            case 3
            groundTruth = GT231;
            case 4
            groundTruth = GT281;
        end
        
        results{descriptor,crossline_idx} = seismicValidate(classifiedImage, groundTruth);
        

    end
    
end 

save('./LabelingResults/LabelingResults.mat','results'); 

%%
for i=1:numDescriptors  
fprintf(option{i}) 
fprintf('  PA  | O_IU   C_IU   F_IU   S_IU    MIU   FWIU\n')
    for j=1:length(crosslineNumber)
        temp = results{i,j}; 
        PA(j) = temp.PA; 
        O_IU(j) = temp.IU(1); 
        C_IU(j) = temp.IU(2); 
        F_IU(j) = temp.IU(3); 
        S_IU(j) = temp.IU(4); 
        MIU(j) = temp.MIU;
        FWIU(j) = temp.FWIU; 
    end
fprintf('%0.4f|%0.4f %0.4f %0.4f %0.4f %0.4f %0.4f\n', mean(PA),mean(O_IU), mean(C_IU), mean(F_IU), mean(S_IU), mean(MIU), mean(FWIU))
    
end


%% 
% crosslineNumber = [61,211,231,281];
% 
% for crossline_idx = 1:4
%         img = squeeze(Crossline(crosslineNumber(crossline_idx),:,:));
%         img = img(105:end, 1:end); % crop TOP 105 PIXELS (noise)
% 
%     switch crossline_idx
%         case 1
%         groundTruth = GT61;
%         case 2
%         groundTruth = GT211;
%         case 3
%         groundTruth = GT231;
%         case 4
%         groundTruth = GT281;
%     end
% 
%     Transperency =  uint8(zeros([size(img),3]));
%     [height, width] = size(groundTruth);
%     for i = 1:height
%         for j = 1:width
%             if groundTruth(i,j) == 2 % CHaotic
%                 Transperency(i,j,:) = [0,0,255]; % blue
%             elseif groundTruth(i,j) == 1 % other:
%                 Transperency(i,j,:) = [184,184,184]; % Gray
%             elseif groundTruth(i,j) == 3  % Fault
%                 Transperency(i,j,:) = [0,255,0]; % Green
%             elseif groundTruth(i,j) == 4 % salt
%                 Transperency(i,j,:) = [255,0,0]; % red
%             end
%         end
%     end
%     
%     name = strcat('.\LabelingResults\Crossline_GT',num2str(crosslineNumber(crossline_idx)), '.bmp');
%     a = 0.45; 
%     coloredImage = (a*im2double(Transperency))+((1-a)*img); 
%     imwrite(coloredImage, name)
%     
%    
%     name = strcat('.\LabelingResults\Crossline_',num2str(crosslineNumber(crossline_idx)), '.bmp');
%     imwrite(img, name)
% end 
