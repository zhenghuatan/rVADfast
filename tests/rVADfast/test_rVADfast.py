import numpy as np

from rVADfast import rVADfast


def test_rvadfast_returns_frame_labels_and_timestamps():
    sampling_rate = 1000
    signal = np.sin(2 * np.pi * 100 * np.arange(sampling_rate) / sampling_rate)

    labels, timestamps = rVADfast()(signal, sampling_rate)

    assert labels.dtype == np.int64
    assert len(labels) == 99
    assert np.array_equal(timestamps, np.arange(len(labels)) * 0.01)


def test_rvadfast_honors_configured_frame_shift():
    signal = np.ones(1000)
    vad = rVADfast(window_duration=0.02, shift_duration=0.02, n_fft=64)

    labels, timestamps = vad(signal, 1000)

    assert len(labels) == 50
    assert np.array_equal(timestamps, np.arange(50) * 0.02)
