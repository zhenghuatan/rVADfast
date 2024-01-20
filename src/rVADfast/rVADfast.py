import math

import numpy as np
from scipy.signal import lfilter

from rVADfast import speechproc


# Refs:
#  [1] Z.-H. Tan, A.k. Sarkara and N. Dehak, "rVAD: an unsupervised segment-based robust voice activity detection method," Computer Speech and Language, 2019. 
#  [2] Z.-H. Tan and B. Lindberg, "Low-complexity variable frame rate analysis for speech recognition and voice activity detection." 
#  IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.

# 2017-12-02, Achintya Kumar Sarkar and Zheng-Hua Tan

class rVADfast:
    def __init__(self, window_duration: float = 0.025, shift_duration: float = 0.01,
                 n_fft: int = 512, sft_threshold: float = 0.5, vad_threshold: float = 0.4,
                 energy_floor: float = np.exp(-50)):
        self.window_duration = window_duration
        self.shift_duration = shift_duration
        self.n_fft = n_fft
        self.sft_threshold = sft_threshold
        self.vad_threshold = vad_threshold
        self.energy_floor = energy_floor

    def __call__(self, signal, sampling_rate):
        frame_length = math.floor(sampling_rate * self.window_duration)
        frame_shift = math.floor(sampling_rate * self.shift_duration)
        n_frames = speechproc.compute_n_frames(signal_length=len(signal),
                                               frame_length=frame_length,
                                               frame_shift=frame_shift)

        # Compute spectral flatness
        sft = speechproc.sflux(signal, frame_length, frame_shift, self.n_fft)

        # Thresholding based on sft value
        pitch_voiced = np.less_equal(sft, self.sft_threshold)

        # High-pass filtering
        b = np.array([0.9770, -0.9770])
        a = np.array([1.0000, -0.9540])
        filtered_signal = lfilter(b, a, signal, axis=0)

        # Segment using SNR weighted energy difference and classify high-energy noise segments
        energy_floor = self.energy_floor
        high_energy = speechproc.snre_highenergy(filtered_signal, n_frames, frame_length,
                                                 frame_shift, energy_floor, pitch_voiced)
        high_energy_segments = np.flatnonzero(np.diff(np.r_[0, high_energy, 0]) != 0).reshape(-1, 2) - [0, 1]
        noise_label = np.zeros_like(high_energy)
        for segment_start, segment_end in high_energy_segments:
            noise_label[segment_start: segment_end] = np.sum(pitch_voiced[segment_start: segment_end]) <= 2

        # Set high-energy noise segments to zero
        noise_label_interpolated = np.interp(np.arange(len(filtered_signal)) / len(filtered_signal),
                                             np.arange(len(noise_label)) / len(noise_label),
                                             noise_label) == 1
        filtered_signal[noise_label_interpolated] = 0

        # Speech enhancement
        # TODO: Should SE be included?

        # VAD based on SNR weighted energy difference
        vad_labels = speechproc.snre_vad(filtered_signal, n_frames, frame_length, frame_shift, energy_floor,
                                         pitch_voiced, self.vad_threshold)

        vad_timestamps = np.arange(len(vad_labels)) * self.shift_duration
        return vad_labels, vad_timestamps
