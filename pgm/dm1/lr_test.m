lr;

% Train set 
n_train = size(data,1);
Y_train_predicted = zeros(n_train,1);

for i = 1:n_train
    if p(data(i,1),data(i,2),w)>0.5
        Y_train_predicted(i,1) = 1;
    end
end

precision_train = sum(data(:,3) ==Y_train_predicted)/n_train;



% Test set
n_test = size(data_test,1);
Y_test_predicted = zeros(n_test,1);

for i = 1:n_test
    if p(data_test(i,1),data_test(i,2),w)>0.5
        Y_test_predicted(i,1) = 1;
    end
end

precision_test = sum(data_test(:,3) == Y_test_predicted)/n_test;

% Output the precisions
precision_train
precision_test


function y = p(u,v,w)
l = [u;v;1];
y = w.'*l;
end