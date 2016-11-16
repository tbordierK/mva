function [rew,draws]=UCB1(T,MAB)

a = size(MAB);
a = a(2);

N = zeros(a,T);
S = zeros(a,T);
mu = zeros(a,T);
draws = zeros(T);

for t=1:T
    if t<K
        S(a,t) = S(a,t) + MAB{t}.sample();
        N(a,t) =  N(a,t) + 1;
        mu(a,t) = S(a,t)/N(a,t);
        draws(t) = t;
    else
        
        [~,arm_to_pull] = max(mu(:,t)+sqrt(log(t)/(2*N(:,t))));
        draws(t) = arm_to_pull;
        S(arm_to_pull,t) = S(arm_to_pull,t) + MAB{arm_to_pull}.sample();
        N(arm_to_pull,t) =  N(arm_to_pull,t) + 1;
        mu(arm_to_pull,t) = S(arm_to_pull,t)/N(arm_to_pull,t);
        
    end
end


end
        