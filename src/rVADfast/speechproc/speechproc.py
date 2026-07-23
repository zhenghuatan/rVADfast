import numpy as np
import math
from copy import deepcopy


SNR_SMOOTHING_RADIUS = 18
LONG_PITCH_EXTENSION = (33, 47)
SHORT_PITCH_EXTENSION = (5, 12)
MIN_SEGMENT_ENERGY = 0.001


# References
# Z.-H. Tan and B. Lindberg, Low-complexity variable frame rate analysis for speech recognition and voice activity detection.
# IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.
# Achintya Kumar Sarkar and Zheng-Hua Tan 2017
# Version: 02 Dec 2017


def compute_n_frames(signal_length, frame_length, frame_shift):
    return math.ceil((signal_length - frame_length) / frame_shift) + 1


def enframe(speech, frame_length, frame_shift, return_padding: bool = False):
    input_length = len(speech)

    if input_length < frame_length:
        raise ValueError("speech file length shorter than window length")

    n_frames = compute_n_frames(signal_length=len(speech), frame_length=frame_length, frame_shift=frame_shift)

    min_signal_length = (n_frames - 1) * frame_shift + frame_length
    padding = min_signal_length - input_length

    if len(speech) < min_signal_length:
        signal = np.concatenate((speech, np.zeros(padding)))

    else:
        signal = deepcopy(speech)

    # Create array for selecting frames of size frame_length
    idx = np.tile(np.arange(0, frame_length), (n_frames, 1))

    # Add frame_shift*frame_no to indexes
    idx = idx + np.tile(np.arange(0, n_frames * frame_shift, frame_shift), (frame_length, 1)).T
    if return_padding:
        return signal[idx], padding
    return signal[idx]


def spectral_flatness(signal, frame_length, n_fft):
    eps = np.finfo(float).eps

    # Create hamming window
    window = np.hamming(frame_length)

    # Frame data
    framed_data = signal * window  # apply window to frame data

    # Magnitude spectrogram
    ak = np.abs(np.fft.fft(framed_data, n_fft))
    idx = range(0, math.floor(n_fft / 2) + 1)
    ak = ak[:, idx]

    # Compute spectral flatness from magnitude spectrogram
    numerator = np.exp(float(1 / len(idx)) * np.sum(np.log(ak + eps), axis=1))
    denominator = float(1 / len(idx)) * np.sum(ak, axis=1)

    return (numerator + eps) / (denominator + eps)


def sflux(signal, frame_length, frame_shift, n_fft):
    framed_data = enframe(signal, frame_length, frame_shift)  # framing
    s_flatness = spectral_flatness(framed_data, frame_length, n_fft)  # compute spectral flatness
    n_frames = compute_n_frames(signal_length=len(signal), frame_length=frame_length, frame_shift=frame_shift)
    # Syn frames as per n_frames
    if n_frames < len(s_flatness):
        s_flatness = s_flatness[:n_frames]
    else:
        s_flatness = np.concatenate((s_flatness, np.repeat(s_flatness[-1], n_frames - len(s_flatness), axis=0)))
    return s_flatness


def estimate_energy(signal, frame_length, frame_shift, energy_floor):
    # Create frames of the signal
    frames = enframe(signal, frame_length, frame_shift)
    # Compute total energy of each frame
    energy = np.sum(np.square(frames), axis=-1)
    # Set parts with energy below energy floor to floor value
    energy[np.less_equal(energy, energy_floor)] = energy_floor
    return energy


def segmentwise_percentile(energy, segment_length, percentile: int = 10):
    energy_segmented = enframe(energy, segment_length, segment_length)
    energy_segmented[energy_segmented == 0] = np.nan
    energy_segmented_min = np.nanpercentile(energy_segmented, percentile, axis=-1)
    return energy_segmented_min


def segmentwise_exponential_smooth(energy, segment_length):
    energy_segmented_min = segmentwise_percentile(energy, segment_length, percentile=10)
    energy_segmented_min_smoothed = np.copy(energy_segmented_min)
    energy_min_smoothed = np.copy(energy)
    n_full_segments = len(energy_segmented_min)
    energy_segmented_min_smoothed[0] = 0.1 * energy_segmented_min[0]
    energy_min_smoothed[0: segment_length] = energy_segmented_min[0]
    for i in range(1, n_full_segments):
        energy_segmented_min_smoothed[i] = 0.9 * energy_segmented_min_smoothed[i - 1] + 0.1 * energy_segmented_min[i]
        energy_min_smoothed[i * segment_length: (i + 1) * segment_length] = energy_segmented_min[i]
    energy_segmented_min[-1] = 0.9 * energy_segmented_min[-2] + 0.1 * energy_segmented_min[-1]
    energy_min_smoothed[n_full_segments * segment_length:] = energy_segmented_min[-1]
    return energy_min_smoothed


def segmentwise_max(signal, segment_length):
    segments = enframe(signal, segment_length, segment_length)
    segments[segments == 0] = np.nan
    segments_max = np.nanmax(segments, axis=-1)
    # Set each segment to segment-wise max
    signal_max = np.copy(signal)
    n_full_segments = len(segments)
    for i in range(n_full_segments):
        signal_max[i * segment_length: (i + 1) * segment_length] = segments_max[i]
    signal_max[n_full_segments * segment_length:] = segments_max[-1]
    return signal_max


def compute_posteriori_snr(energy, energy_min):
    return 10 * (np.log10(energy) - np.log10(energy_min))

def compute_snr_weighted_energy_diff(energy, energy_min):
    posteriori_snr = compute_posteriori_snr(energy, energy_min)
    posteriori_snr = posteriori_snr * (posteriori_snr > 0)
    snr_weighted_energy_diff = np.sqrt(np.abs(energy[1:] - energy[:-1]) * posteriori_snr[1:])
    snr_weighted_energy_diff = np.insert(snr_weighted_energy_diff, 0, snr_weighted_energy_diff[1])
    return snr_weighted_energy_diff

def snre_highenergy(signal, n_frames, frame_length, frame_shift, energy_floor, pitch_voiced):
    segment_threshold_factor = 0.25

    # energy estimation
    energy = estimate_energy(signal, frame_length, frame_shift, energy_floor)

    # Estimation of noise energy
    segment_length = 200
    if n_frames <= segment_length:
        segment_length = n_frames
        energy_min_smoothed = np.nanpercentile(energy, 10, axis=-1)
    else:
        energy_min_smoothed = segmentwise_exponential_smooth(energy, segment_length)

    # Compute a posteriori SNR weighted energy difference
    snr_weighted_energy_diff = compute_snr_weighted_energy_diff(energy, energy_min_smoothed)

    # Central smoothing a posteriori SNR weighted energy difference
    kernel_size = 18 * 2 + 1
    kernel = np.ones(kernel_size) / kernel_size
    snr_weighted_energy_diff_smoothed = np.convolve(snr_weighted_energy_diff, kernel, mode="same")

    # Find segment-wise max and set each segment to segment-wise max
    snr_weighted_energy_diff_smoothed_max = segmentwise_max(snr_weighted_energy_diff_smoothed, segment_length)

    # Classify frames as high-energy frame if smoothed a posteriori SNR weighted energy difference above threshold
    high_energy = np.greater(snr_weighted_energy_diff_smoothed,
                             snr_weighted_energy_diff_smoothed_max * segment_threshold_factor)

    return high_energy


def snre_vad(signal, n_frames, frame_length, frame_shift, energy_floor, pitch_voiced, vad_threshold):
    """Detect voiced frames from SNR-weighted energy changes."""
    pitch_voiced = np.asarray(pitch_voiced, dtype=bool)
    if len(pitch_voiced) != n_frames:
        raise ValueError("pitch_voiced length must match n_frames")

    pitch_voiced_block = pitch_block_detect(pitch_voiced, n_frames)
    energy = estimate_energy(signal, frame_length, frame_shift, energy_floor)
    vad = np.zeros(n_frames, dtype=bool)

    def runs(labels):
        boundaries = np.flatnonzero(np.diff(np.r_[False, labels, False]))
        return boundaries.reshape(-1, 2)

    for start, end in runs(pitch_voiced_block):
        stop = end - 1
        segment_energy = energy[start:stop + 1]
        if len(segment_energy) == 1:
            vad[start:stop + 1] = False
            continue
        energy_min = np.percentile(segment_energy, 10)
        posteriori_snr = np.maximum(np.log10(segment_energy) - np.log10(energy_min), 0)
        energy_difference = np.zeros_like(segment_energy)
        energy_difference[1:] = np.sqrt(
            np.abs(np.diff(segment_energy)) * posteriori_snr[1:])
        energy_difference[0] = energy_difference[1]

        # Boxcar-smooth the energy difference across neighboring frames.
        smoothed_difference = np.convolve(
            np.pad(energy_difference, SNR_SMOOTHING_RADIUS, mode="edge"),
            np.ones(2 * SNR_SMOOTHING_RADIUS + 1),
            mode="valid")[:len(segment_energy)]
        pitch_segment = pitch_voiced[start:stop + 1]
        if np.any(pitch_segment):
            threshold = smoothed_difference[pitch_segment].mean() * vad_threshold
            vad[start:stop + 1] = smoothed_difference > threshold

    initial_vad = vad.copy()
    for start, end in runs(initial_vad):
        stop = end - 1
        pitch_indices = np.flatnonzero(pitch_voiced[start:stop + 1]) + start
        if not len(pitch_indices):
            vad[start:stop + 1] = False
            continue
        first_pitch, last_pitch = pitch_indices[[0, -1]]
        left_extension, right_extension = LONG_PITCH_EXTENSION
        vad[start:max(first_pitch - left_extension, start)] = False
        vad[min(last_pitch + right_extension + 1, stop + 1):stop + 1] = False

    for start, end in runs(initial_vad):
        stop = end - 1
        pitch_indices = np.flatnonzero(pitch_voiced[start:stop + 1]) + start
        if len(pitch_indices) > 4:
            first_pitch, last_pitch = pitch_indices[[0, -1]]
            left_extension, right_extension = SHORT_PITCH_EXTENSION
            vad[max(first_pitch - left_extension, start):first_pitch + 1] = True
            vad[last_pitch + 1:min(last_pitch + right_extension + 1, stop + 1)] = True
        if energy[start:stop + 1].mean() < MIN_SEGMENT_ENERGY:
            vad[start:stop + 1] = False
        if len(pitch_indices) <= 2:
            vad[start:stop + 1] = False

    return vad.astype(np.int64)


def pitch_block_detect(pitch_voiced, n_frames, extension: int = 60):
    # Extended pitch segment detection
    sign_pitch_voiced = 0
    pitch_voiced_block = np.copy(pitch_voiced)
    for i in range(len(pitch_voiced)):
        if (pitch_voiced[i] == 1) and (sign_pitch_voiced == 0):
            n_start, sign_pitch_voiced = i, 1
            pitch_voiced_block[range(max(n_start - extension, 0), n_start + 1)] = True
        elif ((pitch_voiced[i] == 0) or (i == n_frames - 1)) and (sign_pitch_voiced == 1):
            n_stop, sign_pitch_voiced = i, 0
            pitch_voiced_block[range(n_stop, min(n_stop + extension, n_frames - 1) + 1)] = True

    return pitch_voiced_block
