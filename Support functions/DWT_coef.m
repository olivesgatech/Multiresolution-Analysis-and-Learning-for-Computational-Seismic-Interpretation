function [C] = DWT_coef(img,scales)
%this function returns the discrete wavelet for the input
%image (img).
% INPUTS: ================================================
% Img: input image 
% Scales: Number of decomposition scales 

% OUTPUT: ================================================
% C{i}{1}: Approximation coefficients at scale i 
% C{i}{2}: Horizontal coefficients at scale i
% C{i}{3}: Vertical coefficients at scale i
% C{i}{4}: Diagonal coefficients at scale i


% Img: input image 
% Scales: Number of decomposition scales 
% NumBins: number of quantization bins


% WAVELET                   NAME
% ===================================
% Haar              		haar           
% Daubechies        		db             
% Symlets           		sym            
% Coiflets          		coif           
% BiorSplines       		bior           
% ReverseBior       		rbio           
% Meyer             		meyr           
% DMeyer            		dmey           
% Gaussian          		gaus           
% Mexican_hat       		mexh           
% Morlet            		morl           
% Complex Gaussian  		cgau           
% Shannon           		shan           
% Frequency B-Spline		fbsp           
% Complex Morlet    		cmor           
% ===================================

%% 
wname = 'haar'; 
CA = img; 
% C{1}{j}: Approximation coefficients at scale j 
% C{2}{j}: Horizontal coefficients at scale j
% C{3}{j}: Vertical coefficients at scale j
% C{4}{j}: Diagonal coefficients at scale j
[c,s] = wavedec2(img,scales,'haar');

C{1}{1} = appcoef2(c,s,'haar',scales); 
for i=1:scales
[C{2}{i},C{3}{i},C{4}{scales-i+1}] = detcoef2('all',c,s,i);
end 


end