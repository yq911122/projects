function [P,fitvalue,acc_train,acc_test,W,R,C,gamma] = GenPop(popnum,size,x,y,xt,yt)
%UNTITLED5 Summary of this function goes here
%   随机生成初代样本
size=size+8;
P=randi(2,popnum,size)-ones(popnum,size);
for i=1:popnum
    [fitvalue(i),acc_train(i),acc_test(i),W(i),R(i),C(i),gamma(i)]=Fitness(P(i,:),x,y,xt,yt);
end
[fitvalue,idx]=sort(fitvalue,'descend');
P=P(idx,:);
acc_train=acc_train(idx);
acc_test=acc_test(idx);
W=W(idx);
R=R(idx);
C=C(idx);
gamma=gamma(idx);
end

