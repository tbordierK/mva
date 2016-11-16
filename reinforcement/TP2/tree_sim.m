
function [next_state, next_reward] = tree_sim(state, action,max_height,growth,maintenance_cost,planting_cost,wood_price,prob_sick)
% Encoding actions
% 1: not cut/ 2: cut

% States {1,...,Max_height}+{sick:Max_height+1}

    sick_bool = (rand(1)<prob_sick);
    
    if state < max_height
        p = rand();
        idx = p<cumsum(growth(state,:));
        new_growth = find(idx);
    else
        new_growth = 0;
    end
    
    % No cut
    if action == 1

        if state < (max_height+1)
            if sick_bool
                next_state = max_height+1;
            else
                next_state = min(state+new_growth(1),max_height);
            end
            next_reward = -maintenance_cost;
        else
            next_state = state;
            next_reward = 0;
        end
    end

    % Cut
    if action == 2
        if state == max_height+1
            next_state = 1;
            next_reward = -planting_cost;
        else
            next_reward = state*wood_price-planting_cost;
            next_state = 1;
        end
    end
    
    
end


