function [position_matrix, foot_placement, head_velocity] = extract_foot_placement(time_vector, x_position, y_position, body_xposition, head_marker)
Ts = 1/80;
avg_vel = mean(head_marker);
if avg_vel>0
    bool_up=1;
else
    bool_up=0;
end
diff_leg_x  = squeeze(x_position - body_xposition);
foot_placement = 0;
head_velocity = 0;


[~, max_leg] = findpeaks(diff_leg_x, 'MinPeakDistance', 10);
[~, min_leg] = findpeaks(-diff_leg_x, 'MinPeakDistance', 10);
position_matrix = zeros(2,length(time_vector));
bool_contact = zeros(length(time_vector),1);
bool_contact(max_leg) = 1;
bool_contact(min_leg) = -1;

for time = 1 : size(x_position,3)
    if bool_up
        cdt1 = 1; cdt2 = -1;
    else
        cdt1 = -1; cdt2 = 1;
    end
    len_before_time = length(bool_contact(1:time-1));
    len_after_time = length(bool_contact(time:end));
    if bool_contact(time)==cdt1
        idx_next_toeoff = find(bool_contact(time:end)==cdt2,1);
        if ~isempty(idx_next_toeoff)
            idx_endc = idx_next_toeoff;
        else
            idx_endc = length(bool_contact(time:end));
        end
        x_value = mean(x_position(1,1,time:time+idx_endc-1));
        y_value = mean(y_position(1,1,time:time+idx_endc-1));
        position_matrix(1,time:time+idx_endc-1) = x_value;
        position_matrix(2,time:time+idx_endc-1) = y_value;
    end
end



end
