function [X,Y] = GenSim( num, gen_base,gen_add )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Len_b=length(gen_base);
Len_a=length(gen_add);
num=num/2;

Gen_base=repmat(gen_base',num,1);

r1=0.8+0.4*rand(num,1);
R1=repmat(r1,1,Len_b);
X1=R1.*Gen_base+0.002*randn(num,Len_b);

startpt=randi([600,900],num,1);
r=0.8+0.4*rand(num,1);
randplus=@(i)(X1(i,:)+[zeros(1,startpt(i)-1),r(i)*gen_add',zeros(1,Len_b-startpt(i)-Len_a+1)]);
I=1:num;
X1=arrayfun(randplus,I,'UniformOutput', false);
X1=cell2mat(X1');

r2=0.8+0.4*rand(num,1);
R2=repmat(r2,1,Len_b);
X2=R2.*Gen_base+0.002*randn(num,Len_b);

X=[X1;X2];
Y=[-1*ones(num,1);ones(num,1)];

%X=WaveForm(X,6);

%[X,Y]=Randperm(X,Y);

end

function [Xnew,Ynew] = Randperm( X,Y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

A=randperm(length(Y));

C=num2cell(X,2);
C2=C(A);
Xnew=cell2mat(C2);

Ynew=Y(A);
end

