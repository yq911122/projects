function  [val,acc_train,acc_test,W,R,C,gamma]  = Fitness( sig,x,y,xt,yt)
%UNTITLED Summary of this function goes here
%   ������Ӧ��
n=length(x(1,:));

C=bin2dec(sprintf('%s',num2str(sig(n+1:n+4))))+2;
gamma=bin2dec(sprintf('%s',num2str(sig(n+5:end))));
C=2^C;
gamma=2^gamma;
    
R=1+1/C;
W=100000;
%���ݸ���ѡ�����ݼ�����Ӧ����
t=0;
for i=1:length(x(1,:))
    if sig(i)==0
        x(:,i-t)=[];
        xt(:,i-t)=[];
        t=t+1;
    end
end

if isempty(x)
    val=0;
    acc_train=0.5;
    acc_test=0.5;

else

    %train model
    model=svmtrain(y,x,sprintf('-c %f -g %d',C,gamma));

    %��ȡ��ѵ���������Լ��ϵ�׼ȷ��
    [~, acc_train, ~] = svmpredict(y, x, model);
    [~, acc_test, ~] = svmpredict(yt, xt, model);

    acc_train=acc_train(1);
    acc_test=acc_test(1);

    if model.totalSV== length(x(:,1)) %����
        val=0;
    else
        %find R
        %[~,R]=FindBeta(x,2^gamma);%ͨ�������ι滮��ȡRֵ
        R=FindBeta2(x,gamma,C);%ͨ����beta����Ϊ1/n����Rֵ
        %find W        
        W=Findw(model,C);
        
        K1=1;
        K2=0;
        val=K1*1/(W*R)+K2*acc_train;
    end

end
end


function  w  = Findw(model,C)

H=bsxfun(@(U,V) exp(-model.Parameters(4) .* pdist2(U,V,'euclidean').^2),model.SVs,model.SVs)+eye(model.totalSV)/C;
B=model.sv_coef*model.sv_coef';
W=H.*B;

w=sum(W(:));   
    
end

function R = FindBeta2( x,gamma,C)
%UNTITLED3 Summary of this function goes here
%   ���Ƽ���Rֵ
n=length(x(:,1));

H=bsxfun(@(U,V) exp(-gamma .* pdist2(U,V,'euclidean').^2),x,x)+eye(n)/C;

R=1-(sum(H(:)))/(n^2)+1/C;

end

