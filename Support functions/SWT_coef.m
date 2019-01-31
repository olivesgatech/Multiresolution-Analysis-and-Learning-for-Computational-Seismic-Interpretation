function [C] = SWT_coef(img,scales)
%this function returns the stationary wavelet for the input
%image (img).
% INPUTS: ================================================
% Img: input image 
% Scales: Number of decomposition scales 

% OUTPUT: ================================================
% C{i}{1}: Approximation coefficients at scale i
% C{i}{2}: Horizontal coefficients at scale i
% C{i}{3}: Vertical coefficients at scale i
% C{i}{4}: Diagonal coefficients at scale i

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Daubechies: 'db1' or 'haar', 'db2', ... ,'db45'
%   Coiflets  : 'coif1', ... ,  'coif5'
%	Fejer-Korovkin: 'fk4', 'fk6', 'fk8', 'fk14', 'fk18', 'fk22'
%   Symlets   : 'sym2' , ... ,  'sym8', ... ,'sym45'
%   Discrete Meyer wavelet: 'dmey'
%   Biorthogonal:
%       'bior1.1', 'bior1.3' , 'bior1.5'
%       'bior2.2', 'bior2.4' , 'bior2.6', 'bior2.8'
%       'bior3.1', 'bior3.3' , 'bior3.5', 'bior3.7'
%       'bior3.9', 'bior4.4' , 'bior5.5', 'bior6.8'.
%   Reverse Biorthogonal: 
%       'rbio1.1', 'rbio1.3' , 'rbio1.5'
%       'rbio2.2', 'rbio2.4' , 'rbio2.6', 'rbio2.8'
%       'rbio3.1', 'rbio3.3' , 'rbio3.5', 'rbio3.7'
%       'rbio3.9', 'rbio4.4' , 'rbio5.5', 'rbio6.8'.
%       
% ===================================

%% 
ImgDim= size(img); 
if size(img,3)~=1
        cform = makecform('srgb2lab'); 
        temp = applycform(img,cform);
        img = im2double(temp(:,:,1)); 
       % fprintf('Only luminance channel was used\n'); 
end 

if log2(size(img,1))~=ceil(log2(size(img,1)))  
    img = wextend('ar','sym',img,2^(ceil(log2(size(img,1))))-size(img,1),'d'); 
    % fprintf('Image has been extended vertically\n'); 
end 

if log2(size(img,2))~=ceil(log2(size(img,2)))  
    img = wextend('ac','sym',img,2^(ceil(log2(size(img,2))))-size(img,2),'r'); 
    %fprintf('Image has been extended horizontally\n'); 
end 

MaxPossibleScale = log2(min([size(img,1),size(img,2)])); 
if scales > MaxPossibleScale
    error('Maximum possible number of scales is %d\n',MaxPossibleScale);
end 
%img = (img-mean2(img))/std2(img); 
wname = 'haar'; 
[CA,CH,CV,CD] = swt2(img,scales,wname); 
 
for i=1:scales   
    C{i}{1} = CA(1:ImgDim(1),1:ImgDim(2),i);
    C{i}{2} = CH(1:ImgDim(1),1:ImgDim(2),i);
    C{i}{3} = CV(1:ImgDim(1),1:ImgDim(2),i);
    C{i}{4} = CD(1:ImgDim(1),1:ImgDim(2),i);
end 

end