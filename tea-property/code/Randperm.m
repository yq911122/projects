function [Xnew,Ynew] = Randperm( X)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Y=[ones(5,1);-1*ones(5,1)];
A=randperm(length(Y));

C=num2cell(X,2);
C2=C(A);
Xnew=cell2mat(C2);

Ynew=Y(A);
end