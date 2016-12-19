function [P_new,fitvalue_new,acc_train,acc_test,W,R,C,gamma] = GenNextPop( P,x,y,xt,yt,fitvalue )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
%��������
gen=length(P(:,1));

P_new=zeros(size(P));
fitvalue_new=zeros(gen,1);
acc_train=zeros(gen,1);
acc_test=zeros(gen,1);
W=zeros(gen,1);
R=zeros(gen,1);
C=zeros(gen,1);
gamma=zeros(gen,1);

%����ÿ�������ڽ���ѡ���б�ѡ�еĸ���
prolist=fit2pro(fitvalue);

%���н���ѡ��
for i=1:gen
    [fa,ma]=GenParents(prolist);
    P_new(i,:)=CrossOver(P(fa,:),P(ma,:));
end

%���б�������
P_new=Mutate(P_new);

%����ÿ���������Ӧ��
for i=1:gen
    [fitvalue_new(i),acc_train(i),acc_test(i),W(i),R(i),C(i),gamma(i)]=Fitness(P_new(i,:),x,y,xt,yt);
end

%��Ӣ����
elite_num=0.1*gen;
[~,idx]=sort(fitvalue_new);
P_new(idx(1:elite_num),:)=P(1:elite_num,:);

[fitvalue_new,idx]=sort(fitvalue_new,'descend');
P_new=P_new(idx,:);
acc_train=acc_train(idx);
acc_test=acc_test(idx);
W=W(idx);
R=R(idx);
C=C(idx);
gamma=gamma(idx);
end

%������Ӧ�ȵĸ���ѡ�񽻲游ĸ
function [fa,ma]=GenParents(prolist)
    pro1=unifrnd (0,1);
    pro2=unifrnd (0,1);
    for i=1:length(prolist)
        if pro1<prolist(i)
            fa=i;
            break;
        end
    end
    for i=1:length(prolist)
        if pro2<prolist(i)
            ma=i;
            break;
        end
    end
end

function prolist = fit2pro(fitvalue)
gen=length(fitvalue);
total_fit=sum(fitvalue);
prolist=zeros(1,gen);

for i=1:gen
    pro=fitvalue(i)/total_fit;
    if i>1
        prolist(i)=prolist(i-1)+pro;
    else
        prolist(i)=pro;
    end
end

end

function ch = CrossOver( sig1,sig2 )
%UNTITLED3 Summary of this function goes here
%   ����ѡ������
n=length(sig1);
l=round((n-3)/10);

%��ĸ��ÿ0.1*n��Ϊ��λ�������ʻ���
for i=1:(n-3)/l
    pro=unifrnd (0,1);
    if pro>0.5
        ch(l*i-(l-1):l*i)=sig1(l*i-(l-1):l*i);
    else
        ch(l*i-(l-1):l*i)=sig2(l*i-(l-1):l*i);
    end
end 

for i=n-2:n
    pro=unifrnd (0,1);
    if pro>0.5
        ch(i)=sig1(i);
    else
        ch(i)=sig2(i);
    end
end 

end

function [P] = Mutate(P)
%UNTITLED4 Summary of this function goes here
%   ��������-��mutate_rate���죬����������Ϊ1
mutate_rate=0.05;

for i=1:length(P(:,1))
    pro=unifrnd (0,1);
    if pro<=mutate_rate
        j=randi(length(P(1,:)));
        P(i,j)=1-P(i,j);
    end
end
end

