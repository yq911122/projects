function [res1,res2] = FindBeta( x,gamma,C)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
opts = optimoptions('quadprog','Algorithm','active-set','MaxIter',4000);
n=length(x(:,1));

H=2*bsxfun(@(U,V) exp(-gamma .* pdist2(U,V,'euclidean').^2),x,x);

f=[];
A=[];
b=[];
beq=1;
Aeq=ones(1,n);
lb=zeros(1,n);
ub=[];
beta0=ones(1,n)*1/n;

[res1,res2]=quadprog(H,f,A,b,Aeq,beq,lb,ub,beta0,opts);
res2=1-res2;
end

