
file = 'classificationC';
data = importdata(strcat('classification_data_HWK1/',file,'.train'));
data_test = importdata(strcat('classification_data_HWK1/',file,'.test'));

X = data(:,1:2);
Y = data(:,3);

n_echantillon = size(X,1);   % size of sample set

% Retrieve index of I_0 = {samples|Y=0} and I_1 = {samples|Y=1}
I_0 = find(Y<1);
I_1 = find(Y==1);

% Defining parameters
w = rand(3,1);
w_ = rand(3,1);

% Adding 1 to account for b
X(:,3) = 1;

% IRLS
while norm(w-w_)>0.01
   w = w_;
   t = n(X,w);
   w_ = w+inv(X.'*diag(t.*(1-t))*X)*X.'*(Y-t);
end

w_;

% Plotting
fg = figure;
scatter(X(I_0,1),X(I_0,2))
hold on
scatter(X(I_1,1),X(I_1,2))
%Plotting p(y=1|x)
syms x y
ezplot(p(x,y,w_)==0.5)
hold off

title(strcat(file,mat2str(w_)));
print(fg,strcat('logr/',file),'-dpdf','-r0')





function y = n(x,w)
y=sig(x*w);
end

function y = sig(x)
y=1./(1+exp(-x));
end

function y = p(u,v,w)
l = [u;v;1];
y = sig(w.'*l);
end