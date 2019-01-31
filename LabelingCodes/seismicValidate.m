function results = seismicValidate(classifiedImage, groundTruth, numClasses)

if nargin<3
    numClasses = 4;
end

%vectorize:
classifiedImage = classifiedImage(:);
groundTruth = groundTruth(:);

% Calculate confusion matrix:
[C,Order] = confusionmat(groundTruth, classifiedImage);
t = sum(C,2); % # of pixels in each class

% pixel accuracy
PA = trace(C)/sum(t);

% mean class accuracy:
CA = diag(C)./t;
MCA = (1/numClasses)*sum(diag(C)./t);

% Mean IU (intersection over unioun)
IU = diag(C)./(t+ sum(C',2)-diag(C)); 
MIU = (1/numClasses)*sum(diag(C)./(t+ sum(C',2)-diag(C)));

% Frequency Weighted IU
FWIU = 1/(sum(t))   *   sum(  (t.*diag(C))   ./   ( t+ sum(C',2)-diag(C)    ));
results.PA = PA; 
results.CA = CA; 
results.MCA = MCA; 
results.IU = IU; 
results.MIU = MIU; 
results.FWIU = FWIU; 
results.classes = [{'Other'}, {'Chaotic'},{'Fault'},{'Salt'}];

end