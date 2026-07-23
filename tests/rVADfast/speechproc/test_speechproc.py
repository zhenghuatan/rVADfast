import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rVADfast import speechproc


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

    result = speechproc.snre_vad(signal, N_FRAMES, FRAME_LENGTH, FRAME_SHIFT,
                                 ENERGY_FLOOR, pitch_voiced, VAD_THRESHOLD)

    expected = np.ones(N_FRAMES, dtype=np.int64)
    # The first energy transition occurs after frame 1.
    expected[:2] = 0
    # Frames 39-40 cross from the first one-valued interval to silence.
    expected[39:41] = 0
    # Frames 79-81 cover the later silent interval.
    expected[79:82] = 0
    assert np.array_equal(result, expected)


def test_snre_vad_handles_short_and_unvoiced_input():
    signal = np.ones(25)

    result = speechproc.snre_vad(signal, 1, FRAME_LENGTH, 10, ENERGY_FLOOR,
                                 np.array([False]), VAD_THRESHOLD)

    assert np.array_equal(result, np.array([0]))


def test_frame_and_energy_utilities():
    signal = np.arange(1, 11, dtype=float)

    assert speechproc.compute_n_frames(10, 4, 3) == 3
    frames, padding = speechproc.enframe(signal, 4, 3, return_padding=True)
    assert padding == 0
    assert np.array_equal(frames, [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
    assert np.array_equal(
        speechproc.estimate_energy(signal, 4, 3, energy_floor=100),
        [100, 126, 294],
    )

    with pytest.raises(ValueError, match="shorter"):
        speechproc.enframe(np.ones(3), 4, 2)


def test_spectral_and_segment_utilities():
    frames = np.ones((2, 4))
    flatness = speechproc.spectral_flatness(frames, 4, 8)
    assert np.allclose(flatness, [0.00081774, 0.00081774])

    signal = np.arange(1, 11, dtype=float)
    assert speechproc.sflux(signal, 4, 3, 8).shape == (3,)
    energy = np.array([1., 2., 3., 4., 5., 6.])
    assert np.array_equal(speechproc.segmentwise_percentile(energy, 3), [1.2, 4.2])
    assert np.array_equal(
        speechproc.segmentwise_exponential_smooth(energy, 3),
        [1.2, 1.2, 1.2, 4.2, 4.2, 4.2],
    )
    assert np.array_equal(speechproc.segmentwise_max(energy, 3), [3, 3, 3, 6, 6, 6])
    assert np.allclose(speechproc.compute_posteriori_snr([10, 100], [1, 10]), [10, 10])
    assert np.allclose(
        speechproc.compute_snr_weighted_energy_diff(np.array([1., 5., 14.]), np.ones(3)),
        [np.sqrt(90 * np.log10(14)), np.sqrt(40 * np.log10(5)), np.sqrt(90 * np.log10(14))],
    )


def test_pitch_and_high_energy_utilities():
    pitch_voiced = np.array([False, False, True, True, False, False])
    assert np.array_equal(
        speechproc.pitch_block_detect(pitch_voiced, len(pitch_voiced), extension=1),
        [False, True, True, True, True, True],
    )
    signal = np.r_[np.zeros(50), np.ones(50), np.zeros(50)]
    result = speechproc.snre_highenergy(
        signal, n_frames=14, frame_length=25, frame_shift=10,
        energy_floor=ENERGY_FLOOR, pitch_voiced=np.zeros(14, dtype=bool),
    )
    assert result.dtype == bool
    assert result.shape == (14,)


def test_signal_processing_parity_with_legacy_code():
    legacy_path = Path(__file__).parents[3] / "src/rVADfast/legacy_files/speechproc.py"
    spec = importlib.util.spec_from_file_location("legacy_speechproc", legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    signal = np.arange(1, 96, dtype=float)
    modern_frames = speechproc.enframe(signal, 25, 10)
    legacy_frames = legacy.enframe(signal, 1000, 0.025, 0.01)
    modern_flux = speechproc.sflux(signal, 25, 10, 64)
    legacy_flux, _, _, _ = legacy.sflux(signal, 1000, 0.025, 0.01, 64)
    pitch_voiced = np.array([False, True, True, False, False, True, False, False])

    assert np.array_equal(modern_frames, legacy_frames)
    assert np.allclose(modern_flux, legacy_flux)
    assert np.array_equal(
        speechproc.pitch_block_detect(pitch_voiced, len(pitch_voiced)),
        legacy.pitchblockdetect(pitch_voiced, np.zeros(len(pitch_voiced)), len(pitch_voiced), 1),
    )
