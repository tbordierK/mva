%% Build your own bandit problem 

K = 6;
p = randi(100,1,K);

Arm1 = armBernoulli(p(1));
Arm2 = armBernoulli(p(2));
Arm3 = armBernoulli(p(3));
Arm4 = armBernoulli(p(4));
Arm5 = armBernoulli(p(5));
Arm6 = armBernoulli(p(6));

MAB={Arm1,Arm2,Arm3,Arm4,Arm5,Arm6};

% bandit : set of arms

NbArms=length(MAB);

Means=zeros(1,NbArms);
for i=1:NbArms
    Means(i)=MAB{i}.mean;
end

% Display the means of your bandit (to find the best)
Means
muMax=max(Means);


%% Comparison of the regret on one run of the bandit algorithm

T=5000; % horizon

[rew1,draws1]=UCB(T,MAB);
reg1=muMax*(1:T) - cumsum(rew1);
[rew2,draws2]=TS(T,MAB);
reg2=muMax*(1:T) - cumsum(rew2);


plot(1:T,reg1,1:T,reg2)


%% (Expected) regret curve for UCB and Thompson Sampling


