clear; % clear all current variables.

% load training data
tmp = load('double_moon_train1000.mat');
Xtr = tmp.X';
Ytr = tmp.Y';

% load the validation data
tmp = load('double_moon_val1000.mat');
Xval = tmp.X';
Yval = tmp.Y';

% train fully connected neural network with 1 hidden layer
h  = 3; % number of hidden units, i.e. the dimensionality of the hidden layer
di = 2; % input dimension (2D) -- do not change
do = 1; % output dimension (1D - classification) -- do not change

lrate     = 0.02; % learning rate
nsamples  = length(Ytr);
visualization_step = 1000; % visualize output only these steps

% randomly initialize parameters of the model
Wi = rand(h,di);
bi = rand(h,1);
Wo = rand(1,h);
bo = rand(1,1);

i = randi(nsamples);

X = Xtr(:,i);
Y = Ytr(:,i); % desired output
    
% compute gradient 
[grad_s_Wi, grad_s_Wo, grad_s_bi, grad_s_bo] = gradient_nn(X,Y,Wi,bi,Wo,bo);
[grad_s_Wi_, grad_s_Wo_, grad_s_bi_, grad_s_bo_] = gradient_nn_approx(X,Y,Wi,bi,Wo,bo);

grad_s_Wi
grad_s_Wi_
grad_s_Wo
grad_s_Wo_
grad_s_bi
grad_s_bi_
grad_s_bo
grad_s_bo_
