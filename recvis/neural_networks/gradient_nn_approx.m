function [grad_s_Wi, grad_s_Wo, grad_s_bi, grad_s_bo] = ...
                                gradient_nn_approx(X,Y,Wi,bi,Wo,bo)
h = bo/10000; 
[a,b,s] = nnet_forward_logloss(X,Y,Wi,bi,Wo,bo);

shape = size(Wi);
W_h = zeros(shape(1),shape(2));
grad_s_Wi = zeros(shape(1),shape(2));
for i=1:shape(1)
    for j =1:shape(2)
        W_h(i,j) = 1*h;
        [a,b,s_] = nnet_forward_logloss(X,Y,Wi+W_h,bi,Wo,bo);
        grad_s_Wi(i,j) = (s_-s)/h;
        W_h(i,j) = 0;
    end
end


shape = size(bi);
b_h = zeros(shape(1),shape(2)); 
grad_s_bi = zeros(shape(1),shape(2));
for i=1:shape(1)
    for j =1:shape(2)
        b_h(i,j) = 1*h;
        [a,b,s_] = nnet_forward_logloss(X,Y,Wi,bi+b_h,Wo,bo);
        grad_s_bi(i,j) = (s_-s)/h;
        b_h(i,j) = 0;
    end
end

shape = size(Wo);
W_h = zeros(shape(1),shape(2)); 
grad_s_Wo = zeros(shape(1),shape(2));
for i=1:shape(1)
    for j =1:shape(2)
        W_h(i,j) = 1*h;
        [a,b,s_] = nnet_forward_logloss(X,Y,Wi,bi,Wo+W_h,bo);
        grad_s_Wo(i,j) = (s_-s)/h;
        W_h(i,j) = 0;
    end
end

shape = size(bo);
b_h = zeros(shape(1),shape(2)); 
grad_s_bo = zeros(shape(1),shape(2));
for i=1:shape(1)
    for j =1:shape(2)
        b_h(i,j) = 1*h;
        [a,b,s_] = nnet_forward_logloss(X,Y,Wi,bi,Wo,bo+b_h);
        grad_s_bo(i,j) = (s_-s)/h;
        b_h(i,j) = 0;
    end
end




b_h = ones(shape(1),shape(2))*h; 
[a,b,s_] = nnet_forward_logloss(X,Y,Wi,bi,Wo,bo+b_h);
grad_s_bo = (s_-s)/h;

return
end
                       