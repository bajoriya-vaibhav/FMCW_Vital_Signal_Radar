% clc; clear; close all;

%% Radar Configuration
adc_samples = 113;        % Samples per chirp
chirps_per_frame = 125;   % Chirps per frame
num_rx = 4;               % Number of RX antennas
bytes_per_sample = 4;     % (2 bytes I + 2 bytes Q)
frame_period = 0.1958;   % Active Frame period (s) or 11.75ms
fs_frame = 1 / frame_period; % Frame rate (~5.107 Hz) so 5.1 frames per seconds

%% reading .bin file
bin_filename = '.\aditya_breathrate_1Ghz_1\iqData_Raw_0.bin';
fid = fopen(bin_filename, 'rb');
raw_data = fread(fid, 'uint8=>uint8');
fclose(fid);

%% parsing raw_data into complex IQ samples
raw_data = reshape(raw_data, bytes_per_sample, []);
I = double(typecast(reshape(raw_data(1:2,:), [], 1), 'int16'));
Q = double(typecast(reshape(raw_data(3:4,:), [], 1), 'int16'));
IQ = complex(I, Q);

total_samples = length(IQ);
samples_per_frame = adc_samples * chirps_per_frame * num_rx;
num_frames = floor(total_samples / samples_per_frame);
IQ = IQ(1 : num_frames * samples_per_frame);

% IQ(frame, chirp, rx, adc_sample)
IQ = reshape(IQ, [num_rx, adc_samples, chirps_per_frame, num_frames]);
IQ = permute(IQ, [4, 3, 1, 2]); % (frame, chirp, rx, adc_sample)
disp(['Parsed ', num2str(num_frames), ' frames successfully.']);

%% Step 1: Range FFT
range_profiles = fft(IQ, [], 4);
range_profiles_mag = abs(range_profiles);

%% Step 2: Find strongest reflection bin
avg_profile = squeeze(mean(mean(mean(range_profiles_mag, 1), 2), 3));
[~, target_bin_idx] = max(avg_profile);

%% Step 3: Extracting phase at target bin
target_phase = angle(range_profiles(:,:,:,target_bin_idx));
phase_signal = target_phase(:,:,1); % Use RX0
phase_signal = reshape(phase_signal, [], 1);

%% Step 4: Phase unwrapping and conversion to displacement
unwrapped_phase = unwrap(phase_signal);
wavelength = 3e8 / 77e9; % 77 GHz
displacement = unwrapped_phase * wavelength / (4 * pi);
displacement = reshape(displacement, num_frames, chirps_per_frame);

% Average across chirps per frame
chest_displacement = mean(displacement, 2);
chest_displacement = detrend(chest_displacement); % remove trend
chest_displacement = chest_displacement - mean(chest_displacement); % zero mean

%% Plotting chest displacement

figure;
plot(chest_displacement);
xlabel('Samples');
ylabel('Displacement (m)');
title('Raw Chest Displacement for 1 min');
grid on;

%% Step 5: Peak-to-peak check
ptp_disp = max(chest_displacement) - min(chest_displacement);
disp(['Peak-to-peak chest displacement: ', num2str(ptp_disp, '%.2e'), ' meters']);

if ptp_disp < 1e-3
    disp('No significant breathing detected (displacement < 1e-3 m).');
    breathing_cycles = 0;
else
    % Step 6: Filter the signal
    movmean_filter = ones(1, 10) / 10; % Moving average filter
    chest_displacement = conv(chest_displacement, movmean_filter, 'same');
    
    % finding peaks (inhalation points)
    min_peak_distance = 15;
    [peak_values, peak_locs] = findpeaks(chest_displacement, ...
                                        'MinPeakDistance', min_peak_distance);
    
    % finding troughs (exhalation points)
    [trough_values, trough_locs] = findpeaks(-chest_displacement, ...
                                            'MinPeakDistance', min_peak_distance);
    trough_values = -trough_values;
    
    % Combine peaks and troughs and sort them by location
    all_extrema_locs = [peak_locs; trough_locs];
    all_extrema_values = [peak_values; trough_values];
    all_extrema_types = [ones(size(peak_locs)); zeros(size(trough_locs))]; % 1 for peaks, 0 for troughs
    
    [all_extrema_locs, sort_idx] = sort(all_extrema_locs);
    all_extrema_values = all_extrema_values(sort_idx);
    all_extrema_types = all_extrema_types(sort_idx);
    
    % Count complete breathing cycles by ensuring proper alternating pattern
    breathing_cycles = 0;
    i = 1;
    
    while i < length(all_extrema_types)
        current_type = all_extrema_types(i);
        next_type = all_extrema_types(i+1);
        
        % If we have a proper alternation (peak->trough or trough->peak)
        if current_type ~= next_type
            breathing_cycles = breathing_cycles + 0.5; % Half cycle completed
            i = i + 1;
        else
            % Skip duplicate type (two peaks or two troughs in a row)
            i = i + 1;
        end
    end
    
    % Plot the detected peaks and troughs
    figure;
    plot(chest_displacement, 'b-');
    hold on;
    plot(peak_locs, peak_values, 'r^', 'MarkerFaceColor', 'r');
    plot(trough_locs, trough_values, 'mv', 'MarkerFaceColor', 'm');
    
    % Mark complete breathing cycles
    cycle_count = 0;
    for i = 1:length(all_extrema_types)-1
        if all_extrema_types(i) ~= all_extrema_types(i+1)
            midpoint_x = (all_extrema_locs(i) + all_extrema_locs(i+1)) / 2;
            midpoint_y = (all_extrema_values(i) + all_extrema_values(i+1)) / 2;
            if all_extrema_types(i) == 1 % If peak->trough (complete cycle)
                cycle_count = cycle_count + 1;
                text(midpoint_x, midpoint_y, ['C' num2str(cycle_count)], 'FontSize', 8);
            end
        end
    end
    
    legend('Filtered Signal', 'Inhalation', 'Exhalation');
    ylabel('Displacement (m)');
    title(['Breathing Pattern Detection: ', num2str(breathing_cycles, '%.2f'), ' BPM']);
    grid on;
    xlabel('Samples');
    
    disp(['Detected ', num2str(breathing_cycles), ' breathing cycles.']);
end

%% Final Output
fprintf('Estimated Breathing Rate = %.2f breaths per minute\n', breathing_cycles);
