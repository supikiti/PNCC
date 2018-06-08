import numpy as np
import scipy
from librosa.core import stft
from librosa import filters


def medium_time_power_calculation(power_stft_signal, M=2):
    medium_time_power = np.zeros(
        shape=(power_stft_signal.shape[0], power_stft_signal.shape[1]))
    power_stft_signal = np.pad(power_stft_signal, [(M, M), (0, 0)], 'constant')

    for i in range(medium_time_power.shape[0]):
        for j in range(medium_time_power.shape[1]):
            medium_time_power[i, j] = sum(
                [1/(2*M + 1) * power_stft_signal[i + k - M, j]
                 for k in range(5)])
    return medium_time_power


def asymmetric_lawpass_filtering(rectified_signal, lm_a=0.999, lm_b=0.5):
    floor_level = np.zeros(
        shape=(rectified_signal.shape[0], rectified_signal.shape[1]))
    floor_level[0, :] = 0.9 * rectified_signal[0, :]
    for m in range(floor_level.shape[0]):
        for l in range(floor_level.shape[1]):
            if (rectified_signal[m, l] >= floor_level[m-1, l]):
                floor_level[m, l] = lm_a * floor_level[m-1, l] + \
                    (1 - lm_a) * rectified_signal[m, l]
            else:
                floor_level[m, l] = lm_b * floor_level[m-1, l] + \
                    (1 - lm_b) * rectified_signal[m, l]
    return floor_level


def halfwave_rectification(subtracted_lower_envelope, th=0):
    for m in range(subtracted_lower_envelope.shape[0]):
        for l in range(subtracted_lower_envelope.shape[1]):
            if (subtracted_lower_envelope[m, l] < th):
                subtracted_lower_envelope[m, l] = 0
    return subtracted_lower_envelope


def temporal_masking(rectified_signal, lam_t=0.85, myu_t=0.2):
    temporal_masked_signal = np.zeros(
        shape=(rectified_signal.shape[0], rectified_signal.shape[1]))
    online_peak_power = np.zeros(
        shape=(rectified_signal.shape[0], rectified_signal.shape[1]))
    temporal_masked_signal[0, :] = rectified_signal[0, :]
    for m in range(rectified_signal.shape[0]):
        for l in range(rectified_signal.shape[1]):
            online_peak_power[m, l] = max(
                lam_t * online_peak_power[m-1, l], rectified_signal[m, l])
            if rectified_signal[m, l] >= lam_t * online_peak_power[m-1, l]:
                temporal_masked_signal[m, l] = rectified_signal[m, l]
            else:
                temporal_masked_signal[m, l] = myu_t * \
                    online_peak_power[m-1, l]

    return temporal_masked_signal


def after_temporal_masking(temporal_masked_signal, floor_level):
    after_tmpmask = np.zeros(
        shape=(temporal_masked_signal.shape[0],
               temporal_masked_signal.shape[1]))
    for m in range(temporal_masked_signal.shape[0]):
        for l in range(temporal_masked_signal.shape[1]):
            after_tmpmask[m, l] = max(
                temporal_masked_signal[m, l], floor_level[m, l])
    return after_tmpmask


def switch_excitation_or_non_excitation(temporal_masked_signal,
                                        floor_level, lower_envelope,
                                        medium_time_power, c=2):
    final_output = np.zeros(
        shape=(temporal_masked_signal.shape[0],
               temporal_masked_signal.shape[1]))
    c = 2
    for m in range(temporal_masked_signal.shape[0]):
        for l in range(temporal_masked_signal.shape[1]):
            if medium_time_power[m, l] >= c * lower_envelope[m, l]:
                final_output[m, l] = temporal_masked_signal[m, l]
            else:
                final_output[m, l] = floor_level[m, l]
    return final_output


def weight_smoothing(final_output, medium_time_power, N=4, L=40):
    spectral_weight_smoothing = np.zeros(
        shape=(final_output.shape[0], final_output.shape[1]))
    for m in range(final_output.shape[0]):
        for l in range(final_output.shape[1]):
            l_1 = max(l - N, 1)
            l_2 = min(l + N, L)
            spectral_weight_smoothing[m, l] = sum(
                [1/(l_2 - l_1 + 1) *
                 (final_output[m, k] / max(medium_time_power[m, k], 0.0001))
                 for k in range(l_1, l_2)])
    return spectral_weight_smoothing


def time_frequency_normalization(power_stft_signal,
                                 spectral_weight_smoothing):
    return power_stft_signal * spectral_weight_smoothing


def mean_power_normalization(transfer_function,
                             final_output, lam_myu=0.999, L=40, k=1):
    myu = np.zeros(shape=(transfer_function.shape[0]))
    myu[0] = 0.0001
    normalized_power = np.zeros(
        shape=(transfer_function.shape[0], transfer_function.shape[1]))
    for m in range(1, transfer_function.shape[0]):
        myu[m] = lam_myu * myu[m - 1] + \
            (1 - lam_myu) / L * \
            sum([transfer_function[m, k] for k in range(0, L-1)])
    for m in range(final_output.shape[0]):
        normalized_power[m, :] = k * transfer_function[m, :] / myu[m]

    return normalized_power


def power_function_nonlinearity(normalized_power, n=15):
    return normalized_power ** 1/n


def pncc(audio_wave, n_fft=1024, sr=16000, window="hamming",
         n_mels=40, n_pncc=13, weight_N=4, power=2, dct=True):

    pre_emphasis_signal = scipy.signal.lfilter([1.0, -0.97], 1, audio_wave)
    stft_pre_emphasis_signal = np.abs(stft(pre_emphasis_signal,
                                      n_fft=n_fft, window=window)) ** power
    mel_filter = np.abs(filters.mel(sr, n_fft=n_fft, n_mels=n_mels)) ** power
    power_stft_signal = np.dot(stft_pre_emphasis_signal.T, mel_filter.T)
    medium_time_power = medium_time_power_calculation(power_stft_signal)
    lower_envelope = asymmetric_lawpass_filtering(
        medium_time_power, 0.999, 0.5)
    subtracted_lower_envelope = medium_time_power - lower_envelope
    rectified_signal = halfwave_rectification(subtracted_lower_envelope)
    floor_level = asymmetric_lawpass_filtering(rectified_signal)
    temporal_masked_signal = temporal_masking(rectified_signal)
    temporal_masked_signal = after_temporal_masking(
        temporal_masked_signal, floor_level)
    
   
    final_output = switch_excitation_or_non_excitation(
        temporal_masked_signal=temporal_masked_signal,
        floor_level=floor_level, lower_envelope=lower_envelope,
        medium_time_power=medium_time_power)
    
    spectral_weight_smoothing = weight_smoothing(
        final_output=final_output,
        medium_time_power=medium_time_power, N=weight_N)
    
    transfer_function = time_frequency_normalization(
        power_stft_signal=power_stft_signal,
        spectral_weight_smoothing=spectral_weight_smoothing)
    
    normalized_power = mean_power_normalization(
        transfer_function, final_output)
    
    power_law_nonlinearity = power_function_nonlinearity(normalized_power)
    
    print(power_law_nonlinearity)
    print(power_law_nonlinearity.shape)
    dct_v = np.dot(filters.dct(
        n_pncc, power_law_nonlinearity.shape[1]), power_law_nonlinearity.T)
    print("pncc.py")
    if dct:
        return dct_v.T
    else:
        return power_law_nonlinearity.T
