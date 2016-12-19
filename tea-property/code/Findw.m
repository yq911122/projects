function  w  = Findw(model,git)

H=bsxfun(@(U,V) exp(-model.Parameters(4) .* pdist2(U,V,'euclidean').^2),model.SVs,model.SVs);
B=model.sv_coef*model.sv_coef';
W=H.*B;

w=sum(W(:));     ;
end