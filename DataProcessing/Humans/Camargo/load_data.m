mydir = pwd;
myFiles = dir(fullfile(mydir,'*.mat'));
str_cell = split(mydir,'\');
local_name = str_cell{6};
tot_data_saved = NaN(1,3,3);
for ii = 1 : length(myFiles)
    if contains(myFiles(ii).name, 'levelground')



        input_data = load(myFiles(ii).name).data;
        markers_matrix = zeros(size(input_data,1),28,3);
        for jj=1:28
            markers_matrix(:,jj,1) = input_data{:,(jj-1)*3+2};
            markers_matrix(:,jj,2) = input_data{:,(jj-1)*3+3};
            markers_matrix(:,jj,3) = input_data{:,(jj-1)*3+4};
        end

        % Transforming the data to only get the pelvis and feet
        my_markers_matrix = zeros(size(input_data,1),3,3);
        my_markers_matrix(:,1,:) = mean(markers_matrix(:,1:4,:),2);
        my_markers_matrix(:,2,:) = markers_matrix(:,25,:);
        my_markers_matrix(:,3,:) = markers_matrix(:,21,:);
        % Compute velocity of all the markers
        markers_velocity = zeros(size(my_markers_matrix));
        dt = 1/200;
        markers_velocity(3:end-2,:,:) = (my_markers_matrix(1:end-4,:,:) - 8*my_markers_matrix(2:end-3,:,:) + 8*my_markers_matrix(4:end-1,:,:) - my_markers_matrix(5:end,:,:))/(12*dt);
        angle_movement = zeros(size(markers_velocity,1),1);
        angle_movement(3:end-2) = atan2(markers_velocity(3:end-2,1,3),markers_velocity(3:end-2,1,1));

        % get the local straight lines
        bool_vector = extractStraightLineSegments(angle_movement);
        idx_begin = find(diff(bool_vector)==1);
        idx_end = find(diff(bool_vector)==-1);
        for chunk = 1 : 2
            avg_angle = mean(angle_movement(idx_begin(chunk):idx_end(chunk)));
            time_vector = idx_begin(chunk):idx_end(chunk);
            rotated_data = rotate_data(my_markers_matrix, avg_angle, time_vector);
            tot_data_saved = cat(1,tot_data_saved,rotated_data,NaN(1,3,3));
        end
        disp(size(tot_data_saved))
    end
end
save(['data_subject_',local_name,'.mat'],'tot_data_saved');
