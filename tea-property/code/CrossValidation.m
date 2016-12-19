function [best_C,best_gamma] = CrossValidation(y,x)
%UNTITLED Summary of this function goes here
%   通过CrossValidation训练模型，找到最优化的参数
%# grid of parameters
%# grid of parameters
folds = 2;
[C,gamma] = meshgrid(-5:2:15, -10:2:10);

%# grid search, and cross-validation
cv_acc = zeros(numel(C),1);
for i=1:numel(C)
    cv_acc(i) = svmtrain(y, x, ...
                    sprintf('-c %d -g %f -v %d',2^C(i), 2^gamma(i), folds));
end

%# pair (C,gamma) with best accuracy
[~,idx] = max(cv_acc);

%# contour plot of paramter selection
%contour(C, gamma, reshape(cv_acc,size(C))), colorbar
%hold on
%plot(C(idx), gamma(idx), 'rx')
%text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...
%    'HorizontalAlign','left', 'VerticalAlign','top')
%hold off
%xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy')

%# now you can train you model using best_C and best_gamma
best_C = 2^C(idx);
best_gamma = 2^gamma(idx);
%# ...
end

