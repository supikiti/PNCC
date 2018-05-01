import numpy as np
import matplotlib.pyplot as plt
import scipy
import librosa
from librosa.core import power_to_db
from librosa.core import stft
from librosa import filters


def medium_time_power_calculation(p_, M=2):
    q_ = np.zeros(shape=(p_.shape[0], p_.shape[1]))
    p_ = np.pad(p_, [(M, M), (0, 0)], 'constant')

    for i in range(q_.shape[0]):
        for j in range(q_.shape[1]):
            q_[i, j] = sum([1/(2*M + 1) * p_[i + k - M, j] for k in range(5)])
    return q_


def asymmetric_lawpass_filtering(q_in, lm_a=0.999, lm_b=0.5):
    q_out = np.zeros(shape=(q_in.shape[0], q_in.shape[1]))
    q_out[0, :] = 0.9 * q_in[0, :]
    for m in range(q_out.shape[0]):
        for l in range(q_out.shape[1]):
            if (q_in[m, l] >= q_out[m-1, l]):
                q_out[m, l] = lm_a * q_out[m-1, l] + (1 - lm_a) * q_in[m, l]
            else:
                q_out[m, l] = lm_b * q_out[m-1, l] + (1 - lm_b) * q_in[m, l]
    return q_out


def halfwave_rectification(pre_q_0, th=0):
    for m in range(pre_q_0.shape[0]):
        for l in range(pre_q_0.shape[1]):
            if (pre_q_0[m, l] < th):
                pre_q_0[m, l] = 0
    return pre_q_0


def temporal_masking(q_o, lam_t=0.85, myu_t=0.2):
    q_th = np.zeros(shape=(q_o.shape[0], q_o.shape[1]))
    q_p = np.zeros(shape=(q_o.shape[0], q_o.shape[1]))
    q_th[0, :] = q_o[0, :]
    for m in range(q_o.shape[0]):
        for l in range(q_o.shape[1]):
            q_p[m, l] = max(lam_t * q_p[m-1, l], q_o[m, l])
            if q_o[m, l] >= lam_t * q_p[m-1, l]:
                q_th[m, l] = q_o[m, l]
            else:
                q_th[m, l] = myu_t * q_p[m-1, l]

    return q_th


def after_temporal_masking(q_th, q_f):
    r_sp = np.zeros(shape=(q_th.shape[0], q_th.shape[1]))
    for m in range(q_th.shape[0]):
        for l in range(q_th.shape[1]):
            r_sp[m, l] = max(q_th[m, l], q_f[m, l])
    return r_sp


def switch_excitation_or_non_excitation(r_sp, q_f, q_le,
                                        q_power_stft_pre_signal, c=2):
    r_ = np.zeros(shape=(r_sp.shape[0], r_sp.shape[1]))
    c = 2
    for m in range(r_sp.shape[0]):
        for l in range(r_sp.shape[1]):
            if q_power_stft_pre_signal[m, l] >= c * q_le[m, l]:
                r_[m, l] = r_sp[m, l]
            else:
                r_[m, l] = q_f[m, l]
    return r_


def weight_smoothing(r_, q_, N=4, L=40):
    s_ = np.zeros(shape=(r_.shape[0], r_.shape[1]))
    for m in range(r_.shape[0]):
        for l in range(r_.shape[1]):
            l_1 = max(l - N, 1)
            l_2 = min(l + N, L)
            s_[m, l] = sum([1/(l_2 - l_1 + 1) * (r_[m, k] / q_[m, k])
                            for k in range(l_1, l_2)])
    return s_


def time_frequency_normalization(p_, s_):
    return p_ * s_


def mean_power_normalization(t_, r_, lam_myu=0.999, L=40, k=1):
    myu = np.zeros(shape=(t_.shape[0]))
    myu[0] = 0.0001
    u_ = np.zeros(shape=(t_.shape[0], t_.shape[1]))
    for m in range(1, t_.shape[0]):
        myu[m] = lam_myu * myu[m - 1] + \
            (1 - lam_myu) / L * sum([t_[m, k] for k in range(0, L-1)])
    for m in range(r_.shape[0]):
        u_[m, :] = k * t_[m, :] / myu[m]

    return u_


def power_function_nonlinearity(u_, n=15):
    return u_ ** (1/n)


def pncc(audio_wave, n_fft=1024, sr=16000, window="hamming",
         n_mels=40):

    pre_emphasis_signal = scipy.signal.lfilter([1.0, -0.97], 1, audio_wave)
    stft_pre_emphasis_signal = np.abs(stft(pre_emphasis_signal,
                                           n_fft=n_fft, window=window)) ** 2
    mel_filter = np.abs(filters.mel(16000, n_fft=n_fft, n_mels=n_mels)) ** 2
    power_stft_pre_signal = np.dot(stft_pre_emphasis_signal.T, mel_filter.T)
    q_power_stft_pre_signal = np.zeros(shape=(power_stft_pre_signal.shape[0],
                                              power_stft_pre_signal.shape[1]))
    q_ = medium_time_power_calculation(power_stft_pre_signal)
    q_le = asymmetric_lawpass_filtering(q_, 0.999, 0.5)
    pre_q_0 = q_ - q_le
    q_0 = halfwave_rectification(pre_q_0)
    q_f = asymmetric_lawpass_filtering(q_0)
    q_th = temporal_masking(q_0)
    r_sp = after_temporal_masking(q_th, q_f)
    r_ = switch_excitation_or_non_excitation(r_sp=r_sp,
                                             q_f=q_f, q_le=q_le,
                                             q_power_stft_pre_signal=q_power_stft_pre_signal)
    s_ = weight_smoothing(r_=r_, q_=q_)
    t_ = time_frequency_normalization(p_=power_stft_pre_signal, s_=s_)
    u_ = mean_power_normalization(t_, r_)
    v_ = power_function_nonlinearity(u_)
    n_pncc = 20
    dct_v = np.dot(filters.dct(n_mfcc, v_.shape[1]), power_to_db(v_.T))
    return dct_v
