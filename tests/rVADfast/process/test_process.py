from pathlib import Path

import numpy as np

from rVADfast.process import process
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
    expected_output = np.concatenate([np.arange(10, 30), np.arange(50, 70)])
    result = trim_from_vad_timestamps(signal, sampling_rate, vad_timestamps)
    assert np.array_equal(result, expected_output)


def test_frame_label_to_start_stop_handles_boundary_segments():
    assert np.array_equal(
        frame_label_to_start_stop(np.array([1, 1, 0, 1, 1])),
        np.array([[0, 2], [1, 4]]),
    )


def test_worker_function_writes_labels_and_trimmed_audio(tmp_path, monkeypatch):
    input_path = tmp_path / "input.wav"
    signal = np.arange(20, dtype=float)
    monkeypatch.setattr(process.audiofile, "read", lambda _: (signal, 10))
    writes = []
    monkeypatch.setattr(process.audiofile, "write", lambda *args: writes.append(args))

    class Vad:
        shift_duration = 0.1

        def __call__(self, _, __):
            return np.array([0, 1, 1, 0]), None

    process.worker_function(input_path, tmp_path / "labels", tmp_path, Vad())
    label_path = tmp_path / "labels" / "input.wav_vad.txt"
    assert label_path.exists()

    process.worker_function(input_path, tmp_path / "trimmed", tmp_path, Vad(), trim_non_speech=True)
    assert writes[0][0] == tmp_path / "trimmed" / "input.wav"
    assert np.array_equal(writes[0][1], np.arange(2))


def test_batch_processors_and_cli_dispatch(tmp_path, monkeypatch):
    input_path = tmp_path / "nested" / "input.wav"
    input_path.parent.mkdir()
    input_path.touch()
    calls = []
    monkeypatch.setattr(process, "worker_function", lambda *args, **kwargs: calls.append((args, kwargs)))
    class Progress:
        def __init__(self, iterable=None, **_):
            self.iterable = iterable

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, _):
            pass

        def __iter__(self):
            return iter(self.iterable)

    monkeypatch.setattr(process, "tqdm", Progress)

    process.rVADfast_single_process(tmp_path, tmp_path / "out")
    assert len(calls) == 1

    class Pool:
        def __init__(self, processes):
            self.processes = processes

        def imap_unordered(self, func, iterable):
            return map(func, iterable)

        def close(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(process.multiprocessing, "Pool", Pool)
    process.rVADfast_multi_process(tmp_path, tmp_path / "out", n_workers=2)
    assert len(calls) == 2

    dispatched = []
    monkeypatch.setattr(process, "rVADfast_single_process", lambda **kwargs: dispatched.append(kwargs))
    process.main(["rVADfast_process", "--root", str(tmp_path), "--n_workers", "0"])
    assert dispatched[0]["root_folder"] == str(tmp_path)
