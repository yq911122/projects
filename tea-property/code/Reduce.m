function [x,xt,gamma,C,sgn ] = Reduce( sig,x,xt)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

nd=length(sig)-8;
n=length(x(1,:));
piece=floor(n/nd);

threshold=60;

t=0;
for i=1:nd-1
    if sig(i)<=threshold
        sgn(i)=0
        x(:,1+(i-t-1)*piece:(i-t)*piece)=[]; 
        xt(:,1+(i-t-1)*piece:(i-t)*piece)=[];
        t=t+1;
    else
        sgn(i)=1;
    end
end

if sig(nd)<=threshold
    sgn(nd)=0;
    x(:,1+(i-t)*piece:end)=[]; 
    xt(:,1+(i-t)*piece:end)=[];
    t=t+1;
else
    sgn(nd)=1;
end

[gamma,C]=getPar(sig(end-7:end));

end

function [gamma,C]=getPar(A)

threshold=60;
for i=1:4
    if A(i)>threshold
        g(i)=1;
    else
        g(i)=0;
    end
end
gamma=bin2dec(sprintf('%s',num2str(g)));

for i=5:8
    if A(i)>threshold
        c(i)=1;
    else
        c(i)=0;
    end
end
C=bin2dec(sprintf('%s',num2str(c)));
end

