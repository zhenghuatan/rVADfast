import numpy as np
from rVADfast.process import frame_label_to_start_stop, trim_from_vad_timestamps

def test_frame_label_to_start_stop():
    labels = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    expected_output = np.array([[1, 3], [4, 6]]).T
    result = frame_label_to_start_stop(labels)
    assert np.array_equal(result, expected_output)

def test_trim_from_vad_timestamps():
    signal = np.arange(1000)
    sampling_rate = 100
    vad_timestamps = np.array([[0.1, 0.3], [0.5, 0.7]]).T
    print(vad_timestamps.shape)
    expected_output = np.concatenate([np.arange(10, 30), np.arange(50, 70)])
    print(expected_output.shape)
    result = trim_from_vad_timestamps(signal, sampling_rate, vad_timestamps)
    print(result.shape)
    assert np.array_equal(result, expected_output)
