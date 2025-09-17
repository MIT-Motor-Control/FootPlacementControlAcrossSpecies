function position_matrix = extract_foot_placement_data(x_position, y_position, local_time, idx_leg, input_data)
% Extract the position information from the raw dataset

switch idx_leg
    case 1
        stance_input = input_data.L1_StanceStart(local_time);
        swing_input = input_data.L1_SwingStart(local_time);
    case 2
        stance_input = input_data.L2_StanceStart(local_time);
        swing_input = input_data.L2_SwingStart(local_time);
    case 3
        stance_input = input_data.L3_StanceStart(local_time);
        swing_input = input_data.L3_SwingStart(local_time);
    case 4
        stance_input = input_data.R1_StanceStart(local_time);
        swing_input = input_data.R1_SwingStart(local_time);
    case 5
        stance_input = input_data.R2_StanceStart(local_time);
        swing_input = input_data.R2_SwingStart(local_time);
    case 6
        stance_input = input_data.R3_StanceStart(local_time);
        swing_input = input_data.R3_SwingStart(local_time);
    otherwise
        print('Error')
end
position_matrix = zeros(length(local_time),2);
% For each stance start, find the next swing start
idx_start_st = find(stance_input==1);
idx_start_sw = find(swing_input==1);
for ii = 1 : length(idx_start_st)
    idx_next_sw = find(idx_start_sw-idx_start_st(ii)>0,1);
    if isempty(idx_next_sw)
        continue
    else
        position_matrix(idx_start_st(ii):idx_start_sw(idx_next_sw)-1,1) = mean(x_position(idx_start_st(ii):idx_start_sw(idx_next_sw)-1));
        position_matrix(idx_start_st(ii):idx_start_sw(idx_next_sw)-1,2) = mean(y_position(idx_start_st(ii):idx_start_sw(idx_next_sw)-1));
    end
end
end
