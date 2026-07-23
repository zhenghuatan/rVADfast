import numpy as np

from rVADfast.speechproc import snre_vad


N_FRAMES = 100
FRAME_LENGTH = 25
FRAME_SHIFT = 20
ENERGY_FLOOR = np.exp(-50)
VAD_THRESHOLD = 0.4


def test_snre_vad_detects_reference_voiced_region():
    signal = np.concatenate((
        np.zeros(400),
        np.ones(500),
        np.zeros(300),
        np.ones(800),
    ))
    pitch_voiced = np.zeros(N_FRAMES, dtype=bool)
    pitch_voiced[20:45] = True
    pitch_voiced[60:90] = True

    result = snre_vad(signal, N_FRAMES, FRAME_LENGTH, FRAME_SHIFT,
                      ENERGY_FLOOR, pitch_voiced, VAD_THRESHOLD)

    expected = np.ones(N_FRAMES, dtype=np.int64)
    # The first energy transition occurs after frame 1.
    expected[:2] = 0
    # The two silent intervals produce non-voiced gaps.
    expected[39:41] = 0
    expected[79:82] = 0
    assert np.array_equal(result, expected)


def test_snre_vad_handles_short_and_unvoiced_input():
    signal = np.ones(25)

    result = snre_vad(signal, 1, FRAME_LENGTH, 10, ENERGY_FLOOR,
                      np.array([False]), VAD_THRESHOLD)

    assert np.array_equal(result, np.array([0]))
