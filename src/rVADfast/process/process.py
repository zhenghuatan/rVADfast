import os
import sys
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing
from typing import Union
from functools import partial

import numpy as np
from tqdm import tqdm
import audiofile

from rVADfast.rVADfast import rVADfast


def frame_label_to_start_stop(labels: np.ndarray):
    # Convert boolean array to integer array
    int_array = np.asarray(labels, dtype=int)

    # Find indices where the array transitions from 0 to 1 or 1 to 0
    transitions = np.diff(int_array)

    # Find the indices where transitions occur from 0 to 1 (start of groups)
    start_indices = np.where(transitions == 1)[0]

    # Find the indices where transitions occur from 1 to 0 (end of groups)
    end_indices = np.where(transitions == -1)[0]

    # Handle the case where the array starts or ends with 1
    if labels[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if labels[-1]:
        end_indices = np.append(end_indices, len(labels) - 1)

    # Combine start and end indices into pairs
    start_stop_indices = np.stack([start_indices, end_indices])

    return start_stop_indices


def trim_from_vad_timestamps(signal, sampling_rate, vad_timestamps):
    vad_start_end = np.floor(vad_timestamps * sampling_rate).astype(int)
    return np.concatenate([signal[start: end] for start, end in vad_start_end.T])


def worker_function(file, save_folder, root_folder, vad, trim_non_speech: bool = False):
    save_path = Path(os.path.join(save_folder, os.path.relpath(file.parent, root_folder)))
    save_path.mkdir(parents=True, exist_ok=True)
    signal, sampling_rate = audiofile.read(file)

    # Convert to mono be averaging channels if multichannel
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=0)

    vad_labels, _ = vad(signal, sampling_rate)
    vad_timestamps = frame_label_to_start_stop(vad_labels) * vad.shift_duration

    if trim_non_speech:
        save_file_name = file.stem.split(".")[0] + file.suffix
        signal_trimmed = trim_from_vad_timestamps(signal, sampling_rate, vad_timestamps)
        audiofile.write(save_path / save_file_name, signal_trimmed, sampling_rate)
    else:
        save_file_name = file.stem.split(".")[0] + "_vad.txt"
        np.savetxt(save_path / save_file_name, vad_timestamps,
                   fmt='%1.3f', header='Speech Start Time [s], Speech End Time [s]', delimiter=',')


def rVADfast_single_process(root_folder, save_folder: str = ".", extension: str = "wav", trim_non_speech: bool = False,
                            **rvad_kwargs):
    vad = rVADfast(**rvad_kwargs)
    print(f"Scanning {root_folder} for files with {extension=}...")
    filepaths = []
    for file in Path(root_folder).rglob("*." + extension):
        filepaths.append(file)
    print(f"Found {len(filepaths)} files.")

    print("Starting VAD.")
    processing_message = "Trimming non-speech segments" if trim_non_speech else "Generating VAD labels"
    with tqdm(total=len(filepaths), desc=processing_message, unit="files") as pbar:
        for file in filepaths:
            worker_function(file, save_folder, root_folder, vad, trim_non_speech)
            pbar.update(1)


def rVADfast_multi_process(root_folder, save_folder: str = ".", extension: str = "wav", n_workers: Union[int, str] = 1,
                           trim_non_speech: bool = False, **rvad_kwargs):
    vad = rVADfast(**rvad_kwargs)

    print(f"Scanning {root_folder} for files with {extension=}...")
    filepaths = []
    for file in Path(root_folder).rglob("*." + extension):
        filepaths.append(file)
    print(f"Found {len(filepaths)} files.")

    print(f"Starting VAD using multiprocessing pool with {n_workers=}.")
    pool = multiprocessing.Pool(processes=n_workers)
    loader_fn = partial(worker_function, save_folder=save_folder, root_folder=root_folder, vad=vad,
                        trim_non_speech=trim_non_speech)

    processing_message = "Trimming non-speech segments" if trim_non_speech else "Generating VAD labels"
    for _ in tqdm(pool.imap_unordered(func=loader_fn, iterable=filepaths), total=len(filepaths),
                  desc=processing_message, unit="files"):
        pass

    pool.close()
    pool.join()


def main(argv=sys.argv):
    parser = ArgumentParser("Script for processing of multiple audio files using rVADfast.")
    parser.add_argument("--root", type=str, required=True, help="Path to audio file folder.")
    parser.add_argument("--save_folder", type=str, required=False,
                        help="Path to folder where VAD labels/trimmed audio files are saved.", default=None)
    parser.add_argument("--ext", type=str, required=False, help="Audio file extension", default="wav")
    parser.add_argument("--n_workers", type=str, required=False,
                        help="Number of workers used for processing files."
                             "If 0 same process is used, otherwise multiprocessing is used.",
                        default=0)
    parser.add_argument("--trim_non_speech", action='store_true',
                        help="If argument is provided, non-speech segments are removed from the processed waveforms "
                             "and the resulting waveform is saved to 'save_folder'. "
                             "Otherwise, the VAD labels will simply be saved to 'save_folder' in .txt files.",
                        default=False)
    parser.add_argument("--window_duration", type=float, required=False,
                        help="Duration of window in seconds.",
                        default=0.025)
    parser.add_argument("--shift_duration", type=float, required=False,
                        help="Duration of window shift in seconds.",
                        default=0.010)
    parser.add_argument("--n_fft", type=int, required=False,
                        help="Number of fft bins to use.",
                        default=512)
    parser.add_argument("--sft_threshold", type=float, required=False,
                        help="Threshold for spectral flatness.",
                        default=0.5)
    parser.add_argument("--vad_threshold", type=float, required=False,
                        help="Threshold for VAD.",
                        default=0.4)
    parser.add_argument("--energy_floor", type=float, required=False,
                        help="Energy floor.",
                        default=np.exp(-50))
    arguments = parser.parse_args(argv[1:])



    if arguments.n_workers == "max":
        n_workers = multiprocessing.cpu_count()
    elif arguments.n_workers.isdigit():
        n_workers = int(arguments.n_workers)
    else:
        raise ValueError("Unsupported argument for number of workers")

    # TODO: Ideally, you should be able to provide settings like window duration and shift duration as kwarg
    # to support non-deafault settings for rVAD.
    if n_workers > 0:
        rVADfast_multi_process(root_folder=arguments.root, save_folder=arguments.save_folder, extension=arguments.ext,
                               n_workers=n_workers,
                               window_duration=arguments.window_duration,
                               shift_duration=arguments.shift_duration,
                               n_fft=arguments.n_fft,
                               sft_threshold=arguments.sft_threshold,
                               vad_threshold=arguments.vad_threshold,
                               energy_floor=arguments.energy_floor,
                               trim_non_speech=arguments.trim_non_speech)
    else:
        rVADfast_single_process(root_folder=arguments.root, save_folder=arguments.save_folder, extension=arguments.ext,
                                window_duration=arguments.window_duration,
                                shift_duration=arguments.shift_duration,
                                n_fft=arguments.n_fft,
                                sft_threshold=arguments.sft_threshold,
                                vad_threshold=arguments.vad_threshold,
                                energy_floor=arguments.energy_floor,
                                trim_non_speech=arguments.trim_non_speech)
    print("Done.")


if __name__ == "__main__":
    main()
