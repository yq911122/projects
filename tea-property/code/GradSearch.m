function grad = GradSearch( gamma,alpha,beta, x,sig)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
RBF_sig = @(U,V) exp(-gamma .* pdist2(sig.*U,sig.*V,'euclidean').^2);
m_k=@(u,v,k)(-2*gamma*(u-v)*(u-v)*sig(k));

m=length(x(1,:));
n=length(alpha);
ALPHA=zeros(1,n*(n-1)/2);
BETA=zeros(1,n*(n-1)/2);
K=zeros(1,n*(n-1)/2);
dK=zeros(m,n*(n-1)/2);

l=1;
for i=1:n
    for j=i+1:n
        ALPHA(l)=alpha(i)*alpha(j);
        BETA(l)=beta(i)*beta(j);
        K(l)=RBF_sig(x(i,:),x(j,:));
        for k=1:m
            dK(k,l)=m_k(x(i,k),x(j,k),k)*K(l);  
        end
        l=l+1;
    end  
end

part3=zeros(1,m);
part4=zeros(1,m);
grad=zeros(1,m);

part1=2*sum(ALPHA.*K)+sum(alpha.^2);
part2=2*sum(BETA.*K)+sum(beta.^2);
for k=1:m
    part3(k)=2*sum(ALPHA.*dK(k,:));
    part4(k)=2*sum(BETA.*dK(k,:));
    grad(k)=part1*part4(k)+part2*part3(k)-2*sum(alpha)*part4(k)-part3(k);
end

end

