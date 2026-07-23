import numpy as np

from rVADfast.speechproc import snre_vad


def test_snre_vad_detects_reference_voiced_region():
    rng = np.random.default_rng(1729)
    signal = rng.normal(size=2_000)
    pitch_voiced = np.zeros(100, dtype=bool)
    pitch_voiced[20:45] = True
    pitch_voiced[60:90] = True

    result = snre_vad(signal, 100, 25, 20, np.exp(-50), pitch_voiced, 0.4)

    assert np.array_equal(result, np.ones(100, dtype=np.int64))


def test_snre_vad_handles_short_and_unvoiced_input():
    signal = np.ones(25)

    result = snre_vad(signal, 1, 25, 10, np.exp(-50), np.array([False]), 0.4)

    assert np.array_equal(result, np.array([0]))
