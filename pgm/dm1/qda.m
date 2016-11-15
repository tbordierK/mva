
file = 'classificationC';
data = importdata(strcat('classification_data_HWK1/',file,'.train'));
data_test = importdata(strcat('classification_data_HWK1/',file,'.test'));
X = data(:,1:2);
Y = data(:,3);

n_echantillon = size(X,1);   % size of sample set

% Retrieve index of I_0 = {samples|Y=0} and I_1 = {samples|Y=1}
I_0 = find(Y<1);
I_1 = find(Y==1);

% MLEstimators
Pi_MLE = sum(Y)/n_echantillon
mu_0_MLE = sum(X(I_0,1:2),1)/size(I_0,1)
mu_1_MLE = sum(X(I_1,1:2),1)/size(I_1,1)
sigma_0_MLE = 1/size(I_0,1)*((X(I_0,1:2)-mu_0_MLE).'*(X(I_0,1:2)-mu_0_MLE))
sigma_1_MLE = 1/size(I_1,1)*((X(I_1,1:2)-mu_1_MLE).'*(X(I_1,1:2)-mu_1_MLE))


% Plotting

fg = figure;
scatter(X(I_0,1),X(I_0,2))

hold on
scatter(X(I_1,1),X(I_1,2))

% Plotting MLE for mus
plot(mu_0_MLE(1) ,mu_0_MLE(2) ,'*')
text(mu_0_MLE(1) ,mu_0_MLE(2) ,'  \leftarrow \mu_0 MLE')
plot(mu_1_MLE(1) ,mu_1_MLE(2) ,'*')
text(mu_1_MLE(1) ,mu_1_MLE(2) ,'  \leftarrow \mu_1 MLE')

%Plotting p(y=1|x)
syms x y
ezplot(p_1(x,y,Pi_MLE,mu_0_MLE,mu_1_MLE,sigma_0_MLE,sigma_1_MLE)==0.5)

hold off

title(file);

print(fg,strcat('qda/',file),'-dpdf','-r0')


function y = p_1(u,v,pi,mu_0,mu_1,sigma_0,sigma_1)
w = [u,v];
e = exp(-1/2*((w-mu_0)*inv(sigma_0)*(w-mu_0).'-(w-mu_1)*inv(sigma_1)*(w-mu_1).'));
y = 1/(1+(1-pi)*e/pi);
end
