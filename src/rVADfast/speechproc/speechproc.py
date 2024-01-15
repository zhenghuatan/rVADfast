import numpy as np
import math
from copy import deepcopy


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
    energy_segmented[energy_segmented == 0] = np.NaN
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
    segments[segments == 0] = np.NaN
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
    # Extended pitch segment detection
    pitch_voiced_block = pitch_block_detect(pitch_voiced, n_frames)

    ## ---*******- important *******
    # here [0] index array element has  not used

    Dexpl, Dexpr = 18, 18
    Dsmth = np.zeros(n_frames, dtype='float64')
    Dsmth = np.insert(Dsmth, 0, 'inf')

    fdata_ = deepcopy(signal)
    pv01_ = deepcopy(pitch_voiced)
    pvblk_ = deepcopy(pitch_voiced_block)

    fdata_ = np.insert(fdata_, 0, 'inf')
    pv01_ = np.insert(pv01_, 0, 'inf')
    pvblk_ = np.insert(pvblk_, 0, 'inf')

    # energy estimation
    energy = estimate_energy(signal, frame_length, frame_shift, energy_floor)
    energy = np.insert(energy, 0, 'inf')  # temp bug fix

    segsnr = np.zeros(n_frames);
    segsnr = np.insert(segsnr, 0, 'inf')
    segsnrsmth = 1
    sign_segsnr = 0
    D = np.zeros(n_frames)
    D = np.insert(D, 0, 'inf')
    posteriori_snr = np.zeros(n_frames, dtype='float64')
    posteriori_snr = np.insert(posteriori_snr, 0, 'inf')
    snre_vad = np.zeros(n_frames)
    snre_vad = np.insert(snre_vad, 0, 'inf')
    sign_pv = 0

    for i in range(1, n_frames + 1):

        if (pvblk_[i] == 1) and (sign_pv == 0):
            nstart = i
            sign_pv = 1

        elif ((pvblk_[i] == 0) or (i == n_frames)) and (sign_pv == 1):

            nstop = i - 1
            if i == n_frames:
                nstop = i
            sign_pv = 0
            datai = fdata_[
                range((nstart - 1) * frame_shift + 1, (nstop - 1) * frame_shift + frame_length - frame_shift + 1)]
            datai = np.insert(datai, 0, 'inf')

            for j in range(nstart, nstop - 1 + 1):  # previously it was for j=nstart:nstop-1
                for h in range(1, frame_length + 1):
                    energy[j] = energy[j] + np.square(datai[(j - nstart) * frame_shift + h])
                if np.less_equal(energy[j], energy_floor):
                    energy[j] = energy_floor

            energy[nstop] = energy[nstop - 1]

            eY = np.sort(energy[range(nstart, nstop + 1)])
            eY = np.insert(eY, 0, 'inf')  # as [0] is discarding

            emin = eY[int(np.floor((nstop - nstart + 1) * 0.1))]

            for j in range(nstart + 1, nstop + 1):

                posteriori_snr[j] = math.log10(energy[j]) - math.log10(emin)

                if np.less(posteriori_snr[j], 0):
                    posteriori_snr[j] = 0

                D[j] = math.sqrt(np.abs(energy[j] - energy[j - 1]) * posteriori_snr[j])

            D[nstart] = D[nstart + 1]

            tm1 = np.hstack((np.ones(Dexpl) * D[nstart], D[range(nstart, nstop + 1)]))
            Dexp = np.hstack((tm1, np.ones(Dexpr) * D[nstop]))

            Dexp = np.insert(Dexp, 0, 'inf')

            for j in range(0, nstop - nstart + 1):
                Dsmth[nstart + j] = sum(Dexp[range(j + 1, j + Dexpl + Dexpr + 1)])

            Dsmth_thres = sum(Dsmth[range(nstart, nstop + 1)] * pv01_[range(nstart, nstop + 1)]) / sum(
                pv01_[range(nstart, nstop + 1)])

            for j in range(nstart, nstop + 1):
                if np.greater(Dsmth[j], Dsmth_thres * vad_threshold):
                    snre_vad[j] = 1

                    #
    pv_vad = deepcopy(snre_vad)

    nexpl = 33
    nexpr = 47  # % 29 and 39, estimated statistically, 95% ; 33, 47 %98 for voicebox pitch
    sign_vad = 0
    for i in range(1, n_frames + 1):
        if (snre_vad[i] == 1) and (sign_vad == 0):
            nstart = i
            sign_vad = 1
        elif ((snre_vad[i] == 0) or (i == n_frames)) and (sign_vad == 1):
            nstop = i - 1
            if i == n_frames:
                nstop = i
            sign_vad = 0
            for j in range(nstart, nstop + 1):
                if pv01_[j] == 1:
                    break

            pv_vad[range(nstart, np.max([j - nexpl - 1, 1]) + 1)] = 0

            for j in range(0, nstop - nstart + 1):
                if pv01_[nstop - j] == 1:
                    break

            pv_vad[range(nstop - j + 1 + nexpr, nstop + 1)] = 0

    nexpl = 5
    nexpr = 12  # ; % 9 and 13, estimated statistically 5%; 5, 12 %2 for voicebox pitch
    sign_vad = 0
    for i in range(1, n_frames + 1):
        if (snre_vad[i] == 1) and (sign_vad == 0):
            nstart = i
            sign_vad = 1
        elif ((snre_vad[i] == 0) or (i == n_frames)) and (sign_vad == 1):
            nstop = i - 1
            if i == n_frames:
                nstop = i
            sign_vad = 0

            if np.greater(sum(pv01_[range(nstart, nstop + 1)]), 4):
                for j in range(nstart, nstop + 1):
                    if pv01_[j] == 1:
                        break

                pv_vad[range(np.maximum(j - nexpl, 1), j - 1 + 1)] = 1
                for j in range(0, nstop - nstart + 1):
                    if pv01_[nstop - j] == 1:
                        break
                pv_vad[range(nstop - j + 1, min(nstop - j + nexpr, n_frames) + 1)] = 1

            esegment = sum(energy[range(nstart, nstop + 1)]) / (nstop - nstart + 1)
            if np.less(esegment, 0.001):
                pv_vad[range(nstart, nstop + 1)] = 0

            if np.less_equal(sum(pv01_[range(nstart, nstop + 1)]), 2):
                pv_vad[range(nstart, nstop + 1)] = 0

    sign_vad = 0
    esum = 0
    for i in range(1, n_frames + 1):
        if (pv_vad[i] == 1) and (sign_vad == 0):
            nstart = i
            sign_vad = 1
        elif ((pv_vad[i] == 0) or (i == n_frames)) and (sign_vad == 1):
            nstop = i - 1
            if i == n_frames:
                nstop = i
            sign_vad = 0
            esum = esum + sum(energy[range(nstart, nstop + 1)])

    #
    eps = np.finfo(float).eps

    eave = esum / (sum(pv_vad[1:len(pv_vad)]) + eps)  # except [0] index 'inf'

    sign_vad = 0
    for i in range(1, n_frames + 1):
        if (pv_vad[i] == 1) and (sign_vad == 0):
            nstart = i
            sign_vad = 1
        elif ((pv_vad[i] == 0) or (i == n_frames)) and (sign_vad == 1):
            nstop = i - 1
            if i == n_frames:
                nstop = i
            sign_vad = 0

            # if np.less(sum(energy[range(nstart,nstop+1)])/(nstop-nstart+1), eave*0.05):
            # pv_vad[range(nstart,nstop+1)] = 0

    #
    sign_vad = 0
    vad_seg = np.zeros((n_frames, 2), dtype="int64")
    n_vad_seg = -1  # for indexing array
    for i in range(1, n_frames + 1):
        if (pv_vad[i] == 1) and (sign_vad == 0):
            nstart = i
            sign_vad = 1
        elif ((pv_vad[i] == 0) or (i == n_frames)) and (sign_vad == 1):
            nstop = i - 1
            sign_vad = 0
            n_vad_seg = n_vad_seg + 1
            # print i, n_vad_seg, nstart, nstop
            vad_seg[n_vad_seg, :] = np.array([nstart, nstop])

    vad_seg = vad_seg[:n_vad_seg + 1, ]

    # syn  from [0] index
    vad_seg = vad_seg - 1

    # print vad_seg

    # make one dimension array of (0/1)
    xYY = np.zeros(n_frames, dtype="int64")
    for i in range(len(vad_seg)):
        k = range(vad_seg[i, 0], vad_seg[i, 1] + 1)
        xYY[k] = 1

    vad_seg = xYY

    return vad_seg


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
