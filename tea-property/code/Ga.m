function Ga(x,y,xt,yt,gen,dataidx,popnum)
%UNTITLED6 Summary of this function goes here

%随机生成初代样本
[P,fitvalue(:,1),acc_train(:,1),acc_test(:,1),M(:,1),R(:,1),C(:,1),gamma(:,1)]=GenPop(popnum,length(x(1,:)),x,y,xt,yt);
T(1,:)=sum(P);
%xlswrite(sprintf('data/%s/P',num2str(dataidx)),P,sprintf('%s',num2str(1)));
%xlswrite(sprintf('data/%s/P',num2str(dataidx)),sum(P),gen+1,sprintf('A%s:AL%s',num2str(1),num2str(1)));

for i=2:gen
     %xlswrite(sprintf('data/%s/P',num2str(dataidx)),P,sprintf('%s',num2str(i)));
     %xlswrite(sprintf('data/%s/P',num2str(dataidx)),sum(P),gen+1,sprintf('A%s:AL%s',num2str(i),num2str(i)));
     [P,fitvalue(:,i),acc_train(:,i),acc_test(:,i),M(:,i),R(:,i),C(:,i),gamma(:,i)]=GenNextPop( P,x,y,xt,yt,fitvalue(:,i-1) );
     T(i,:)=sum(P);
end
     xlswrite(sprintf('data/%s/FIT',num2str(dataidx)),[1:gen]',sprintf('A2:A%s',num2str(gen+1)));
     xlswrite(sprintf('data/%s/FIT',num2str(dataidx)),1:popnum,'B1:GS1');
     xlswrite(sprintf('data/%s/FIT',num2str(dataidx)),fitvalue',sprintf('B%s:GS%s',num2str(2),num2str(gen+1)));
     
     xlswrite(sprintf('data/%s/M',num2str(dataidx)),[1:gen]',sprintf('A2:A%s',num2str(gen+1)));
     xlswrite(sprintf('data/%s/M',num2str(dataidx)),1:popnum,'B1:GS1');
     xlswrite(sprintf('data/%s/M',num2str(dataidx)),M',sprintf('B%s:GS%s',num2str(2),num2str(gen+1)));
     
     xlswrite(sprintf('data/%s/R',num2str(dataidx)),[1:gen]',sprintf('A2:A%s',num2str(gen+1)));
     xlswrite(sprintf('data/%s/R',num2str(dataidx)),1:popnum,'B1:GS1');
     xlswrite(sprintf('data/%s/R',num2str(dataidx)),R',sprintf('B%s:GS%s',num2str(2),num2str(gen+1)));
     
     xlswrite(sprintf('data/%s/ACC_TEST',num2str(dataidx)),[1:gen]',sprintf('A2:A%s',num2str(gen+1)));
     xlswrite(sprintf('data/%s/ACC_TEST',num2str(dataidx)),1:popnum,'B1:GS1');
     xlswrite(sprintf('data/%s/ACC_TEST',num2str(dataidx)),acc_test',sprintf('B%s:GS%s',num2str(2),num2str(gen+1)));
     
     xlswrite(sprintf('data/%s/ACC_TRAIN',num2str(dataidx)),[1:gen]',sprintf('A2:A%s',num2str(gen+1)));
     xlswrite(sprintf('data/%s/ACC_TRAIN',num2str(dataidx)),1:popnum,'B1:GS1');
     xlswrite(sprintf('data/%s/ACC_TRAIN',num2str(dataidx)),acc_train',sprintf('B%s:GS%s',num2str(2),num2str(gen+1)));    
     
     xlswrite(sprintf('data/%s/C',num2str(dataidx)),[1:gen]',sprintf('A2:A%s',num2str(gen+1)));
     xlswrite(sprintf('data/%s/C',num2str(dataidx)),1:popnum,'B1:GS1');
     xlswrite(sprintf('data/%s/C',num2str(dataidx)),C',sprintf('B%s:GS%s',num2str(2),num2str(gen+1)));    
     
     xlswrite(sprintf('data/%s/gamma',num2str(dataidx)),[1:gen]',sprintf('A2:A%s',num2str(gen+1)));
     xlswrite(sprintf('data/%s/gamma',num2str(dataidx)),1:popnum,'B1:GS1');
     xlswrite(sprintf('data/%s/gamma',num2str(dataidx)),gamma',sprintf('B%s:GS%s',num2str(2),num2str(gen+1)));    
     
     xlswrite(sprintf('data/%s/P',num2str(dataidx)),[1:gen]',sprintf('A2:A%s',num2str(gen+1)));
     xlswrite(sprintf('data/%s/P',num2str(dataidx)),1:length(x(1,:))+8,'B1:AM1');
     xlswrite(sprintf('data/%s/P',num2str(dataidx)),T,sprintf('B%s:AM%s',num2str(2),num2str(length(x(1,:))+9)));    
     
end

