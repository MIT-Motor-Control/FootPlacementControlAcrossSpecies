% This is the main extraction script for the mouse dataset. It assumes that the data has been previously segmented in subfiles (using the data_split.m function)

close all; clear all; clc;

% The few lines below characterize the datasets that have to be processed. In this example, only the non mutant cohorts are investigated (i.e. groups 7 and 8 in the original dataset).

StringArray = {"group7_joints1.mat","group7_joints2.mat","group7_joints3.mat","group7_joints4.mat","group7_joints5.mat","group8_joints.mat"};
output_array = {"mat_group71.mat","mat_group72.mat","mat_group73.mat","mat_group74.mat","mat_group75.mat","mat_group8.mat"};
raw_output = {"raw_group71.mat","raw_group72.mat","raw_group73.mat","raw_group74.mat","raw_group75.mat","raw_group8.mat"};
output_times = {"output_time71.mat","output_time72.mat","output_time73.mat","output_time74.mat","output_time75.mat","output_time8.mat"};

% If you want to investigate the mutant cohorts, use the following set of lines (to be uncommented) instead of the previous ones

% StringArray = {"group1_joints.mat","group2_joints.mat","group3_joints.mat","group4_joints.mat","group5_joints.mat","group6_joints.mat"};
% output_array = {"mat_group1_straight.mat","mat_group2_straight.mat","mat_group3_straight.mat","mat_group4_straight.mat","mat_group5_straight.mat",'mat_group6_straight.mat'};
% raw_output = {'raw_group1_straight.mat','raw_group2_straight.mat','raw_group3_straight.mat','raw_group4_straight.mat','raw_group5_straight.mat','raw_group6_straight.mat'};
% output_times = {'output_time1_straight.mat','output_time2_straight.mat','output_time3_straight.mat','output_time4_straight.mat','output_time5_straight.mat','output_time6_straight.mat'};

beha_input = load('allC100_8C_CON.mat').sC1CON; % This data contains the behavioral clusters that allow to identify which sections of the dataset correspond to locomotion
idx_of_interest = [1 7 8 14 15 16]; % We only focus on the nose, base of tail and foot markers, the other ones will be ignored
animal_total = 1;
matrix_metrics = zeros(1,8); % This is an important output file when processing the mutant cohorts. It is also used to characterize the feedforward controller
f_condition = waitbar(0, 'Condition');  
for condition = 1 : length(StringArray)
    tot_foot_contact = NaN(2,1,4); % Output data for the foot placement matrix
    tot_foot_raw = NaN(2,1,length(idx_of_interest)); % Output data for the raw matrix
    tot_time = zeros(1,2); % Output data for the locomotion bout timings
    data_input = load(StringArray{condition}).tmp_struct;
    waitbar(condition/length(StringArray), f_condition, sprintf('Progress through the conditions : %d %%',floor(condition/length(StringArray)*100)));
    f_animal = waitbar(0,'Animal');
    for id_animal = 1 : size(data_input,1)
        waitbar(id_animal/size(data_input,1), f_animal, sprintf('Progress through the animals:  %d %%', floor(id_animal/size(data_input,1)*100)));
        for id_day = 1 : size(data_input,2)
            if condition==6
                condition_b = 8;
            else
                condition_b = 7;
                id_day = id_day+1;
            end
            vec_loco = beha_input{condition_b}{id_animal, id_day};
            matrix_animal = data_input{id_animal, id_day};
            timings_locomotion = get_locomotion_bouts(vec_loco); % Grab the time stamps of the locomotion bouts
            timings_locomotion = timings_locomotion(timings_locomotion(:,2)>50,:); % Grab only the long locomotion bouts
            % For each of the exploitable locomotion bouts, we compute the orientation to determine if this is a straight line, rotate the data to align the main movement direction with the first dimension, extract the foot placement and the associated metrics.
            for line = 1 : size(timings_locomotion,1)
                idx_begin = timings_locomotion(line,1);
                idx_end = timings_locomotion(line,1) + timings_locomotion(line,2)-1;
                local_mat = double(matrix_animal(:,:,idx_begin:idx_end));
                % 1. Computing the body orientation during that locomotion bout (this is done based on the orientation of the nose-tail base axis)
                tmp_orientation = atan2(local_mat(1,2,:)-local_mat(16,2,:), local_mat(1,1,:)-local_mat(16,1,:));
                tmp_orientation(tmp_orientation<0) = tmp_orientation(tmp_orientation<0) + 2*pi;
                tmp_orientation = tmp_orientation * 360 / (2*pi);
                range = max(tmp_orientation) - min(tmp_orientation);
                if abs(range)<30 % If the animal remains relatively straight proceed, otherwise it stops.
                    local_mat = matrix_animal(idx_of_interest,:,idx_begin:idx_end);
                    % 2. Align the main movement direction with the first data dimension
                    tmp_local_x = reshape(local_mat(:,1,:),1,numel(local_mat(:,1,:)));
                    tmp_local_y = reshape(local_mat(:,2,:),1,numel(local_mat(:,2,:)));

                    pca_input = squeeze(local_mat(:,:,1));
                    for ii = 2 : size(local_mat,3)
                        pca_input = [pca_input; squeeze(local_mat(:,:,ii))];
                    end
                    pca_input = double(pca_input) - mean(pca_input,1);
                    [coeff, score, ~, ~, ~, ~] = pca(double(pca_input));
                    local_angle = atan(coeff(2,1)/coeff(1,1));
                    rot_mat = [cos(-local_angle) -sin(-local_angle); sin(-local_angle) cos(-local_angle)];
                    reconstructed_local_mat = zeros(size(local_mat));
                    local_mean_vec = [mean(mean(local_mat(:,1,:))); mean(mean(local_mat(:,2,:)))];
                    for ii = 1 : size(reconstructed_local_mat,1)
                        x_plot = linspace(ii, size(score,1)-(length(idx_of_interest)-ii),timings_locomotion(line,2));
                        reconstructed_local_mat(ii,:,:) = rot_mat * (squeeze(double(local_mat(ii,:,:)))-local_mean_vec) + local_mean_vec;
                    end
                    reconstructed_local_mat_ = zeros(2*size(reconstructed_local_mat,1),size(reconstructed_local_mat,2),size(reconstructed_local_mat,3));
                    reconstructed_local_mat_(1:length(idx_of_interest),:,:) = reconstructed_local_mat;
                    reconstructed_local_mat_(length(idx_of_interest)+1:end,:,3:end-2) = (-reconstructed_local_mat(1:length(idx_of_interest),:,5:end)+8*reconstructed_local_mat(1:length(idx_of_interest),:,4:end-1)-8*reconstructed_local_mat(1:length(idx_of_interest),:,2:end-3)+reconstructed_local_mat(1:length(idx_of_interest),:,1:end-4))/(12*1/80);
                    % 3. Extracting the foot placement
                    local_pos_mat = zeros(2,size(local_mat,3),4);
                    x_position_local = zeros(4,1,size(local_mat,3));
                    y_position_local = zeros(4,1,size(local_mat,3));
                    for leg = 1 : 4
                        x_position_local(leg,:,:) = reconstructed_local_mat_(leg+1,1,:);
                        y_position_local(leg,:,:) = reconstructed_local_mat_(leg+1,2,:);
                        head_marker = reconstructed_local_mat_(8,1,:);
                        vel_marker = sqrt(reconstructed_local_mat_(leg+8,1,:).^2 + reconstructed_local_mat_(leg+8,2,:).^2);
                        [local_pos_mat(:,:,leg),~,~] = extract_foot_placement(x_plot, x_position_local(leg,:,:),y_position_local(leg,:,:), vel_marker, head_marker);
                    end
                    x_position_local(5,:,:) = reconstructed_local_mat_(1,1,:);
                    x_position_local(6,:,:) = reconstructed_local_mat_(6,1,:);
                    x_position_local(7,:,:) = reconstructed_local_mat_(7,1,:);
                    y_position_local(5,:,:) = reconstructed_local_mat_(1,2,:);
                    y_position_local(6,:,:) = reconstructed_local_mat_(6,2,:);
                    y_position_local(7,:,:) = reconstructed_local_mat_(7,2,:);

                    % 4. Extracting the velocity-dependent locomotion metrics (step length, widht, timing, ...)
                    matrix_metrics = [matrix_metrics; extract_metrics(local_pos_mat, idx_begin, idx_end, animal_total, mean(vel_marker(3:end-2)))];
                    tot_foot_contact = [tot_foot_contact, local_pos_mat, NaN(2,1,4)];
                    tot_foot_raw = [tot_foot_raw, [shiftdim(x_position_local,1); shiftdim(y_position_local,1)], NaN(2,1,length(idx_of_interest))];
                    tot_time = [tot_time; [idx_begin,idx_end]];
                end
            end
        end
        animal_total = animal_total+1;

    end
    close(f_animal);
    tot_time = tot_time(2:end,:);
    save(output_times{condition},'tot_time');
    save(output_array{condition},'tot_foot_contact');
    save(raw_output{condition},'tot_foot_raw');
end
save('output_metrics_straight.mat','matrix_metrics');
close(f_condition);
