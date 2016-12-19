function [X,XT] = Scale( X,XT )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
data=[X;XT];
%I=1:length(A(1,:));

%A=arrayfun(@(i)(A(:,i)/norm(A(:,i))),I,'UniformOutput', false);
%A=cell2mat(A);

A=(data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));

X=A(1:length(X(:,1)),:);
XT=A(length(X(:,1))+1:end,:);

end

