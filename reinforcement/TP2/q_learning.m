% Parameters
clearvars;
max_height = 5;
S = max_height+1;
sick_state = max_height+1;
A = 2; % Number of possible actions

rate = 0.05;
gamma = 1/(1+rate);

maintenance_cost = 1;
planting_cost = 2;
wood_price = 5;
prob_sick = 0.1;

growth=zeros(max_height-1,max_height-1);
growth(1:2,1:3)=1/3;
growth(3,1:2)=1/2;
growth(4,1)=1;

Q = zeros(S,A);
% Number of visits for each state
N = zeros(S,A);
eps = 0.1;


T_max = 1000;

for k=1:1000
    x_t = 1;
    for t=0:T_max
        
        % Define what action to perform
        p = rand();
        if p<eps || t==0
            a_t = randi(2);
        else
           [~,a_t] = max(Q(x_t,:));
        end

        [x_t1,r_t] = tree_sim(x_t, a_t,max_height,growth,maintenance_cost,planting_cost,wood_price,prob_sick);
        N(x_t,a_t) = N(x_t,a_t)+1;
        [b,~] = max(Q(x_t1,:));
        Q(x_t,a_t) = (1-alpha(N(x_t,a_t),x_t,a_t))*Q(x_t,a_t)+alpha(N(x_t,a_t),x_t,a_t)*(r_t+gamma*b);
        x_t = x_t1;
    end
end

V_optimal = zeros(S,1);
pi_optimal = zeros(S,1);

for i=1:S
    [V_optimal(i),pi_optimal(i)] = max(Q(i,:));
end

V_optimal