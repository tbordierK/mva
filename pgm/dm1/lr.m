file = 'classificationC';
data = importdata(strcat('classification_data_HWK1/',file,'.train'));
data_test = importdata(strcat('classification_data_HWK1/',file,'.test'));

X = data(:,1:2);
Y = data(:,3);

% Retrieve index of I_0 = {samples|Y=0} and I_1 = {samples|Y=1}
I_0 = find(Y<1);
I_1 = find(Y==1);
X(:,3) = 1;

% Solving normal equation, we suppose here XX.' inversible
w = inv(X.'*X)*X.'*Y;

% Plotting
fg = figure;
scatter(X(I_0,1),X(I_0,2))
hold on
scatter(X(I_1,1),X(I_1,2))
%Plotting p(y=1|x)
syms x y
ezplot(p(x,y,w)==0.5)
hold off

title(strcat(file,mat2str(w)));
print(fg,strcat('lr/',file),'-dpdf','-r0')

function y = p(u,v,w)
l = [u;v;1];
y = w.'*l;
end
