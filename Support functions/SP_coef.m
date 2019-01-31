function [C] = SP_coef(img,scalea,orinetations)
% This function returns the steerable pyramid decomposition for the input
% image (img).

[pyr,pind]=buildSCFpyr(img,scalea,orinetations-1);
C{1} = pyrBand(pyr, pind,1); 
count = 2; 
    for i=2:1:size(pind,1)-1
    C{count} = pyrBand(pyr, pind,i);
    count = count+1; 
    end
C{count} = pyrBand(pyr, pind,scalea*orinetations+2); 
end