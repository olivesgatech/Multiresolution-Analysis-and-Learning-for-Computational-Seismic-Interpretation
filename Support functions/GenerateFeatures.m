function y = GenerateFeatures(I,option)
I = double(I);
scales = 3; 
orientations = 8; 
    switch lower(option)
        case lower('Amplitude') 
            y = amplitude_features(I);

        case lower('GaussianPyr')
            y = gaussian_features(I,scales);

        case lower('Gabor')
            y = Gabor_features(I,scales,orientations);
        case lower('DWT')
            y = DWT_features(I,scales);

        case lower('SWT')
            y = SWT_features(I,scales); 

        case lower('SP')
            y = SP_features(I,scales,orientations);

        case lower('CnT')
            y = CnT_features(I,scales);

        case lower('NCnT')
            y = NCnT_features(I,scales);

        case lower('CT')
            y = CT_features(I);  
        otherwise
            error('Unknown option'); 
    end 

end 

%% Support functions
function Features = amplitude_features(I)
    Features = ExtractSVD(I); 
end 

function Features = gaussian_features(I,scales)
   
    temp = I; 
    f = ExtractSVD(temp); 
        for j=1:scales-1 
        temp = impyramid(temp,'reduce');
        f = [f,ExtractSVD(temp)]; 
        end 
    Features = f;  
end 

function Features = Gabor_features(I,scales,orientations)
    gaborArray = gaborFilterBank(scales,orientations,39,39);
    for i = 1:size(gaborArray,1)
        for j = 1:size(gaborArray,2)
            T{i}{j} = imfilter(double(I), gaborArray{i,j});
        end
    end
            
    f = [];
    for i=1:size(T,2)   
        for k=1:size(T{i},2)
            f = [f,ExtractSVD(T{i}{k})]; 
        end 
    end
    
    Features = f; 
     
end 

function Features = DWT_features(I,scales)
    T = DWT_coef(I,scales); %DWT coefficients
    f = [];
    for j=1:size(T,2)   
        for k=1:size(T{j},2)
            f = [f,ExtractSVD(T{j}{k})]; 
        end 
    end
    Features = f; 

end

function Features = SWT_features(I,scales)

    T = SWT_coef(I,scales); %SWT coefficients
    f = [];
    for j=1:size(T,2)   
        for k=1:size(T{j},2)
        f = [f,ExtractSVD(T{j}{k})];  
        end 
    end
    Features = f; 

end 

function Features = SP_features(I,scales, orientations)
    T = SP_coef(I,scales,orientations); %SP coefficients
    f = [];
    for j=1:length(T)
      f = [f,ExtractSVD(T{j})]; 
    end 
    Features = f;  
end 

function Features = CT_features(I)

    T = fdct_wrapping(I,1); %CT coefficients
    f = [];
    for j=1:size(T,2)
        for k=1:size(T{j},2)
        f = [f,ExtractSVD(T{j}{k})];  
        end 
    end 
    Features = f; 

end 

function Features = NCnT_features(I,scales)
    nlevels = [3,4,4,4] ;        % Decomposition level
    nlevels = nlevels(1:scales);
    T = nsctdec(double(I),nlevels); %NCnT coefficients
    f = [ExtractSVD(T{1})]; %the inner most scale 
    for j=2:size(T,2)
        for k=1:size(T{j},2)
        f = [f,ExtractSVD(T{j}{k})]; 
        end 
    end 
    Features = f; 

end 

function Features = CnT_features(I,scales)
    nlevels = [3,4,4,4] ;        % Decomposition level
    nlevels = nlevels(1:scales);
    pfilter = 'pkva' ;              % Pyramidal filter
    dfilter = 'pkva' ;              % Directional filter
 
    [M, N] = size(I); 
    M_pad = (2^(length(nlevels)+1)*ceil(M/2^(length(nlevels)+1))-M); 
    N_pad = (2^(length(nlevels)+1)*ceil(N/2^(length(nlevels)+1))-N);
    I_ext = padarray(I,[ceil(M_pad/2),ceil(N_pad/2)],'symmetric'); 
    I_ext = I_ext(1:end-mod(M_pad,2),1:end-mod(N_pad,2)); 
            
    T = pdfbdec(double(I_ext), pfilter, dfilter,nlevels); %CnT coefficients
    f = [ExtractSVD(T{1})]; %the inner most scale 
    for j=2:size(T,2)
        for k=1:size(T{j},2)
        f = [f,ExtractSVD(T{j}{k})]; 
        end 
    end 
    Features = f; 

end 


function S = ExtractSVD(X)
option = 1;% Truncate SVD
%option = 0;%(default) no trunaction
S = svd(X,'econ')';

    if sum(S)~=0 && option==1
       w = S/sum(S);
       w(w==0) = []; 
       Q = floor(exp(sum(-w.*log(w)))); 
       S(Q+1:end)= 0; 
    end 
end 