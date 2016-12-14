G = [[2,-1];[0,1]];

n_actions_A = size(G,1);
n_actions_B = size(G,2);
n = 10000;

[actions_A,actions_B,rewards_A] = EXP3vEXP3(n,0.01,0.1,G);

p_a = zeros(n,n_actions_A);
for t=1:n
    for k=1:n_actions_A
        p_a(t,k) = size(find(actions_A(1:t)==k),2)/t;
    end
end

p_b = zeros(n,n_actions_B);
for t=1:n
    for k=1:n_actions_B
        p_b(t,k) = size(find(actions_B(1:t)==k),2)/t;
    end
end


close all
plot(p_a(:,1))
hold on
plot(p_b(:,1))
legend('pa','pb','Location','southwest')
hold off 


figure
cumulative_reward = zeros(n,1);
for t=1:n
    cumulative_reward(t) = sum(rewards_A(1:t))/t;
end

plot(cumulative_reward)