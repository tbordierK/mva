function [grad_s_Wi, grad_s_Wo, grad_s_bi, grad_s_bo] = ...
                                gradient_nn(X,Y,Wi,bi,Wo,bo)
% Compute gradient of the logistic loss of the 
% neural network on example X with target label Y, 
% with respect to the parameters Wi,bi,Wo,bo.
%
% Input: X ... 2x1 vector of the input example
%        Y ... 1x1 the target label in {-1,1}   
%        Wi,bi,Wo,bo ... parameters of the network
%        Wi ... [hxd]
%        bi ... [hx1]
%        Wo ... [1xh]
%        bo ... 1x1
%        where h... is the number of hidden units
%              d... is the number of input dimensions (d=2)
%
% Output: 
%  grad_s_Wi [hxd] ... gradient of loss s(Y,Y(X)) w.r.t  Wi
%  grad_s_Wo [1xh] ... gradient of loss s(Y,Y(X)) w.r.t. Wo
%  grad_s_bi [hx1] ... gradient of loss s(Y,Y(X)) w.r.t. bi
%  grad_s_bo [1x1] ... gradient of loss s(Y,Y(X)) w.r.t. bo
%

% To be completed:

Y_neural = Wo*relu(Wi*X+bi)+bo;
scalar = -Y*exp(-Y*Y_neural)/(1+exp(-Y*Y_neural));

grad_s_Wi =  scalar*Wo'*X'.*((Wi*X+bi)>0);
grad_s_Wo =  scalar*relu(Wi*X+bi)';
grad_s_bi =  scalar*Wo'.*((Wi*X+bi)>0);
grad_s_bo =  scalar;




end

