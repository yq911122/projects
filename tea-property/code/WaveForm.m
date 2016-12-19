function appres = WaveForm( x,level )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

for i=1:length(x(:,1))
    [c,l]=wavedec(x(i,:),level,'db2');
    appres(i,:)=appcoef(c,l,'db2',level);
end
appres=wthresh(appres,'s',0.002);
end

