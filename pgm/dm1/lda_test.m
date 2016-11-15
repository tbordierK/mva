lda;

% Train set 
n_train = size(data,1);
Y_train_predicted = zeros(n_train,1);

for i = 1:n_train
    if p_1(data(i,1),data(i,2),Pi_MLE,mu_0_MLE,mu_1_MLE,sigma_MLE)>0.5
        Y_train_predicted(i,1) = 1;
    end
end

precision_train = sum(data(:,3) ==Y_train_predicted)/n_train;



% Test set
n_test = size(data_test,1);
Y_test_predicted = zeros(n_test,1);

for i = 1:n_test
    if p_1(data_test(i,1),data_test(i,2),Pi_MLE,mu_0_MLE,mu_1_MLE,sigma_MLE)>0.5
        Y_test_predicted(i,1) = 1;
    end
end

precision_test = sum(data_test(:,3) ==Y_test_predicted)/n_test;

% Output the precisions
precision_train
precision_test


function y = p_1(u,v,pi,mu_0,mu_1,sigma)
w = [u,v];
e = exp(-1/2*((w-mu_0)*inv(sigma)*(w-mu_0).'-(w-mu_1)*inv(sigma)*(w-mu_1).'));
y = 1/(1+(1-pi)*e/pi);
end