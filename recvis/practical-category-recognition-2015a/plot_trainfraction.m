% For Hellinger Kernel

figure
x = [0.1,0.5,1];
y = [0.59, 0.73,0.77];
plot(x,y)

hold on


x = [0.1,0.5,1];
y = [0.53,0.64,0.71];
plot(x,y)
x = [0.1,0.5,1];
y = [0.37, 0.54,0.63];
plot(x,y)

title('AP for Hellinger kernel')
xlabel('Fraction of training data used') % x-axis label
ylabel('AP') % y-axis label
legend('Person','Aeroplanes','Motorbikes','Location','southwest')
axis([0 1 0 1])
hold off



% For the linear kernel


figure
x = [0.1,0.5,1];
y = [0.60, 0.67,0.71];
plot(x,y)

hold on


x = [0.1,0.5,1];
y = [0.32,0.40,0.55];
plot(x,y)

x = [0.1,0.5,1];
y = [0.25, 0.38,0.48];
plot(x,y)

title('AP for Linear kernel')
xlabel('Fraction of training data used') % x-axis label
ylabel('AP') % y-axis label
legend('Person','Aeroplanes','Motorbikes','Location','southwest')
axis([0 1 0 1])
hold off