clc; clear; close all;

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
    breathing_rate_bpm = 0;
else
    %% Step 6: FFT for breathing rate estimation
    movmean_filter = ones(1, 10) / 10; % Moving average filter
    chest_displacement = conv(chest_displacement, movmean_filter, 'same');
    
    nfft = 4096;
    f = (0:nfft-1) * (fs_frame / nfft);
    breath_spectrum = abs(fft(chest_displacement, nfft));

    % Focus on a reasonable breathing frequency range
    f_low = 0.11; f_high = 0.7; % Hz
    valid_idx = (f >= f_low) & (f <= f_high);
    f_valid = f(valid_idx);
    spectrum_valid = breath_spectrum(valid_idx);

    % Find peak
    [~, idx_peak] = max(spectrum_valid);
    breathing_freq_hz = f_valid(idx_peak);
    breathing_rate_bpm = breathing_freq_hz * 60;

    % Plot breathing frequency spectrum
    figure;
    plot(f_valid, spectrum_valid);
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    title('Breathing Frequency Spectrum');
    grid on;
    hold on;
    plot(breathing_freq_hz, spectrum_valid(idx_peak), 'ro', 'MarkerSize', 10);
    text(breathing_freq_hz, spectrum_valid(idx_peak), ...
         sprintf(' %.2f Hz (%.1f BPM)', breathing_freq_hz, breathing_rate_bpm));
    hold off;
end

%% Final Output
fprintf('Estimated Breathing Rate = %.2f breaths per minute\n', breathing_rate_bpm);
