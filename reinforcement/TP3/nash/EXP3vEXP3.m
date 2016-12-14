function [ActionsA,ActionsB,Rew]= EXP3vEXP3(n,eta,beta,G)

ActionsA = zeros(1,n);
ActionsB = zeros(1,n);
Rew = zeros(1,n);

n_actions_A = size(G,1);
n_actions_B = size(G,2);

weights_A = ones(1,n_actions_A);
weights_B = ones(1,n_actions_B);

 for i=1:n
    p_A = zeros(1,n_actions_A);
    s = sum(weights_A(:));
    for k=1:n_actions_A
        p_A(k) = (1-beta)*weights_A(k)/s + beta/n_actions_A;
    end
    %weights_A
    %p_A
    arm_to_pull_A = find(mnrnd(1,p_A));
    ActionsA(i) = arm_to_pull_A;
    
    p_B = zeros(1,n_actions_B);
    s = sum(weights_B(:));
    for k=1:n_actions_B
        p_B(k) = (1-beta)*weights_B(k)/s + beta/n_actions_B;
    end
    arm_to_pull_B = find(mnrnd(1,p_B));
    ActionsB(i) = arm_to_pull_B;
    
    
    
    Rew(i) = G(arm_to_pull_A,arm_to_pull_B);
    
    estimated_reward_A = Rew(i)/p_A(arm_to_pull_A);
    estimated_reward_B = -Rew(i)/p_B(arm_to_pull_B);
    
    weights_A(arm_to_pull_A) = weights_A(arm_to_pull_A)*exp(eta*estimated_reward_A);
    weights_B(arm_to_pull_B) = weights_B(arm_to_pull_B)*exp(eta*estimated_reward_B);
    
 end   
 

    
    
    