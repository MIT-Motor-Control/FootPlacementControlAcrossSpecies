
close all; clc; clear all;
input_data = load('Data\20181025_20180530-20180614_IsoD1_Glass_MaskedModel_1000PCs_amplitude_phase_down_downcam_Steps(_down_cam).mat').newData;
input_fields = ["Frame","FlyID","videoID","Orient","xCOM","yCOM","head_x","head_y","L1_xCam","L1_yCam","L2_xCam","L2_yCam","L3_xCam","L3_yCam","R1_xCam","R1_yCam","R2_xCam","R2_yCam","R3_xCam","R3_yCam"];
truncated_list = input_data{:,input_fields};
body_length = input_data{:,"BodyLength"};
truncated_list = rescale_data_to_mm(truncated_list);


%%
padded_list = NaN(1,size(truncated_list,2));
padded_length = NaN(1,1);
padded_time = NaN(1,1);
idx_diff = find((truncated_list(2:end,2) ~= truncated_list(1:end-1,2)));
for line = length(idx_diff):-1:2
    disp(line)
    padded_list = [NaN(1,size(truncated_list,2)); truncated_list(idx_diff(line-1)+1:idx_diff(line),:);padded_list];
    padded_length = [NaN(1,1); body_length(idx_diff(line-1)+1:idx_diff(line)); padded_length];
    padded_time = [NaN(1,1); (idx_diff(line-1)+1:idx_diff(line))'; padded_time];
end

padded_list = [NaN(1,size(truncated_list,2)); truncated_list(1:idx_diff(1),:); padded_list];
padded_length = [NaN(1,1); body_length(1:idx_diff(1),1); padded_length];
padded_time = [NaN(1,1); (1:idx_diff(1))'; padded_time];
% padded_list = normalize_padded_list(padded_list, padded_length);
%%
com_velocity = zeros(size(padded_list,1),2);
com_velocity(3:end-2,:) = (-padded_list(5:end,[5,6]) + 8 * padded_list(4:end-1,[5,6]) - 8 * padded_list(2:end-3,[5,6]) + padded_list(1:end-4,[5,6])) / 12;
tot_com_velocity = sqrt(com_velocity(:,1).^2 + com_velocity(:,2).^2);

%%
local_data = load('Data\20181025_20180530-20180614_IsoD1_Glass_MaskedModel_1000PCs_amplitude_phase_down_downcam_Steps(_down_cam).mat').newData;
clc;
idx_x = [7,9,11,13,15,17,19]; idx_y = [8,10,12,14,16,18,20];
idx_nan = find(isnan(padded_list(:,1)));
vec_message = zeros(length(idx_nan)-1,1);
reconstructed_padded_matrix = padded_list;
tmp_x_matrix = [reconstructed_padded_matrix(:,9) reconstructed_padded_matrix(:,11) reconstructed_padded_matrix(:,13) reconstructed_padded_matrix(:,15) reconstructed_padded_matrix(:,17) reconstructed_padded_matrix(:,19)];
tmp_y_matrix = [reconstructed_padded_matrix(:,10) reconstructed_padded_matrix(:,12) reconstructed_padded_matrix(:,14) reconstructed_padded_matrix(:,16) reconstructed_padded_matrix(:,18) reconstructed_padded_matrix(:,20)];
padded_contacts = cat(3,tmp_x_matrix,tmp_y_matrix);
padded_contacts_data = cat(3,tmp_x_matrix,tmp_y_matrix);%zeros(size(tmp_x_matrix,1),6,2);
[b,a] = butter(6, 45/75, 'low');
offset = 1;
for nan_value = 1 : length(idx_nan)-1
    disp(nan_value)
    reconstructed_padded_matrix(idx_nan(nan_value),:) = NaN(1,size(reconstructed_padded_matrix,2));
    local_mat = padded_list(idx_nan(nan_value)+1:idx_nan(nan_value+1)-1,:);
    local_pos_mat = padded_contacts(idx_nan(nan_value)+1:idx_nan(nan_value+1)-1,:);
    pca_input = reshape(squeeze(local_mat(:,idx_x)),1,[]);
    pca_input = [pca_input; reshape(squeeze(local_mat(:,idx_y)),1,[])];
    pca_input = pca_input - mean(pca_input,2);
    [coeff, score, ~, ~, ~, ~] = pca(pca_input');
    local_angle = atan(coeff(2,1)/coeff(1,1));
    rot_mat = [cos(-local_angle) -sin(-local_angle); sin(-local_angle) cos(-local_angle)];
    local_mean_x = mean(mean(local_mat(:,idx_x)));
    local_mean_y = mean(mean(local_mat(:,idx_y)));
    for point = 1 : 8
        for time = 1 : size(local_mat,1)
            reconstructed_padded_matrix(time+offset,4) = reconstructed_padded_matrix(time+offset, 4) - mean(reconstructed_padded_matrix(idx_nan(nan_value)+1:idx_nan(nan_value+1)-1,4));
            reconstructed_padded_matrix(time+offset,[2*point+3, 2*point+4]) = rot_mat * squeeze([local_mat(time,2*point+3)-local_mean_x local_mat(time,2*point+4)-local_mean_y])' + [local_mean_x, local_mean_y]';
        end
    end
    vec_message(nan_value) = get_message(padded_list, tot_com_velocity, nan_value);
    offset = offset + time + 1;
    x_position_local = zeros(6,size(local_mat,1));
    y_position_local = zeros(6,size(local_mat,1));
    for leg = 1 : 6
        local_len = idx_nan(nan_value+1) - idx_nan(nan_value);
        local_time_vector = padded_time(idx_nan(nan_value)+1:idx_nan(nan_value+1)-1);
        if local_len > 20
            x_position_local(leg,:) = filtfilt(b,a,reconstructed_padded_matrix(idx_nan(nan_value)+1:idx_nan(nan_value+1)-1,7+2*leg));
            y_position_local(leg,:) = filtfilt(b,a,reconstructed_padded_matrix(idx_nan(nan_value)+1:idx_nan(nan_value+1)-1,8+2*leg));
        else
            x_position_local(leg,:) = reconstructed_padded_matrix(idx_nan(nan_value)+1:idx_nan(nan_value+1)-1,7+2*leg);
            y_position_local(leg,:) = reconstructed_padded_matrix(idx_nan(nan_value)+1:idx_nan(nan_value+1)-1,8+2*leg);
        end
        vel_marker = compute_velocity(x_position_local(leg,:));
        [padded_contacts_data(idx_nan(nan_value)+1:idx_nan(nan_value+1)-1,leg,:)] = extract_foot_placement_data(x_position_local(leg,:), y_position_local(leg,:), local_time_vector, leg, local_data);
        [padded_contacts(idx_nan(nan_value)+1:idx_nan(nan_value+1)-1,leg,:),~,~] = extract_foot_placement_vel(linspace(1,size(local_mat,1),size(local_mat,1)), x_position_local(leg,:), y_position_local(leg,:), vel_marker);
    end
end
%% We want to extract good subsegments as well ...
tot_keep = zeros(length(padded_time),1);
for nan_value = 1 : length(idx_nan)-1
    if (vec_message(nan_value)~=4)
        idx_begin = idx_nan(nan_value)+1; idx_end = idx_nan(nan_value+1)-1;
        local_com_velocity = tot_com_velocity(idx_begin:idx_end);
        bool_fast = (local_com_velocity>0.005); % Find fast segments
        local_keep = zeros(length(bool_fast),1);
        count = 0;
        for ii = 1 : length(bool_fast)-1
            if ((bool_fast(ii)==1) && (count==0) && (length(bool_fast)-ii > 49))
                next_bad = find(bool_fast(ii+1:end)==0,1);
                tv = idx_begin+ii:idx_begin+ii+next_bad-1;
                sub_message = get_sub_message(padded_list,tv);
                if ((next_bad > 49) && (sub_message==1))
                    local_keep(tv-idx_begin) = 1;
                    count = count + next_bad -1;
                    % We also have to rotate the corresponding data
                    % using the similar PCA approach as we did above
                    local_mat = padded_list(tv,:);
                    local_pos_mat = padded_contacts_data(tv,:);
                    pca_input = reshape(squeeze(local_mat(:,idx_x)),1,[]);
                    pca_input = [pca_input; reshape(squeeze(local_mat(:,idx_y)),1,[])];
                    pca_input = pca_input - mean(pca_input,2);
                    [coeff, score, ~, ~, ~, ~] = pca(pca_input');
                    local_angle = atan(coeff(2,1)/coeff(1,1));
                    rot_mat = [cos(-local_angle) -sin(-local_angle); sin(-local_angle) cos(-local_angle)];
                    local_mean_x = mean(mean(local_mat(:,idx_x)));
                    local_mean_y = mean(mean(local_mat(:,idx_y)));
                    for point = 1 : 8
                        for time = 1 : size(local_mat,1)
                            padded_list(idx_begin+ii+time-1,4) = padded_list(tv(time),4) - rad2deg(local_angle);
                            padded_list(idx_begin+ii+time-1,[2*point+3, 2*point+4]) = rot_mat * squeeze([local_mat(time,2*point+3)-local_mean_x local_mat(time,2*point+4)-local_mean_y])' + [local_mean_x, local_mean_y]';
                        end
                    end
                    for leg = 1 : 6
                        for time = 1 : size(local_mat,1)
                            if (padded_contacts_data(idx_begin+ii+time-1,leg,1)~=0)
                                padded_contacts_data(idx_begin+ii+time-1,leg,:) = rot_mat * squeeze([padded_contacts_data(idx_begin+ii+time-1,leg,1)-local_mean_x padded_contacts_data(idx_begin+ii+time-1,leg,2)-local_mean_y])' + [local_mean_x, local_mean_y]';
                            end
                        end
                    end

                end
            end
            if count~=0
                count = count-1;
            end
        end
        tot_keep(idx_begin:idx_end) = local_keep;
    end
end


%% Saving the different variables for further analyses
save('Data/ProcessedData/validity_mm.mat','vec_message');
save('Data/ProcessedData/raw_pos_mm.mat','reconstructed_padded_matrix');
save('Data/ProcessedData/contact_data_mm.mat','padded_contacts');
save('Data/ProcessedData/contact_data_mm_fixed.mat','padded_contacts_data');
% save('Data/ProcessedData/tot_keep_vector_mm_fixed.mat','tot_keep');
disp('Data saved you can processed to the next steps :-) ')