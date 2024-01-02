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

def worker_function(file, save_folder, root_folder, vad):
    save_file_name = file.stem.split(".")[0] + ".txt"
    save_path = Path(os.path.join(save_folder, os.path.relpath(file.parent, root_folder)))
    save_path.mkdir(parents=True, exist_ok=True)
    signal, sampling_rate = audiofile.read(file)
    vad_labels, _ = vad(signal, sampling_rate)
    #TODO: Ideally, you should be able to provide settings like window duration and shift duration
    # to support non-deafault settings for rVAD.
    vad_timestamps = frame_label_to_start_stop(vad_labels)*vad.shift_duration
    np.savetxt(save_path / save_file_name, vad_timestamps.astype(int),
               fmt='%d', header='Speech Start Time, Speech End Time', delimiter=',')


def rVAD_single_process(root_folder, save_folder: str = ".", extension: str = "wav"):
    vad = rVADfast()
    print(f"Scanning {root_folder} for files with {extension=}")
    filepaths = []
    for file in Path(root_folder).rglob("*." + extension):
        filepaths.append(file)
    print(f"Found {len(filepaths)} files.")

    print("Starting VAD process")
    with tqdm(total=len(filepaths), desc="Generating VAD labels", unit="files") as pbar:
        for file in filepaths:
            worker_function(file, save_folder, root_folder, vad)
            pbar.update(1)


def rVAD_multi_process(root_folder, save_folder: str = ".", extension: str = "wav", n_workers: Union[int, str] = 1):
    vad = rVADfast()
    print(f"Scanning {root_folder} for files with {extension=}")
    filepaths = []
    for file in Path(root_folder).rglob("*." + extension):
        filepaths.append(file)
    print(f"Found {len(filepaths)} files.")

    print(f"Starting VAD multiprocessing pool with {n_workers=}")
    pool = multiprocessing.Pool(processes=n_workers)
    loader_fn = partial(worker_function, save_folder=save_folder, root_folder=root_folder, vad=vad)
    for _ in tqdm(pool.imap_unordered(func=loader_fn, iterable=filepaths), total=len(filepaths),
                  desc="Generating VAD labels", unit="files"):
        pass

    pool.close()
    pool.join()


def main(argv=sys.argv):
    parser = ArgumentParser("Script for processing of multiple audio files using rVAD fast.")
    parser.add_argument("--root", type=str, required=True, help="Path to audio file folder.")
    parser.add_argument("--save_folder", type=str, required=False, help="Path to folder where VAD labels are saved.",
                        default=None)
    parser.add_argument("--ext", type=str, required=False, help="Audio file extension", default="wav")
    parser.add_argument("--n_workers", type=str, required=False,
                        help="Number of workers used for processing files."
                             "If 0 same process is used, otherwise multiprocessing is used.",
                        default=0)
    arguments = parser.parse_args(argv[1:])

    if arguments.n_workers == "max":
        n_workers = multiprocessing.cpu_count()
    elif arguments.n_workers.isdigit():
        n_workers = int(arguments.n_workers)
    else:
        raise ValueError("Unsupported argument for number of workers")

    if n_workers > 0:
        rVAD_multi_process(root_folder=arguments.root, save_folder=arguments.save_folder, extension=arguments.ext,
                           n_workers=n_workers)
    else:
        rVAD_single_process(root_folder=arguments.root, save_folder=arguments.save_folder, extension=arguments.ext)
    print("Done.")


if __name__ == "__main__":
    main()
    print("done")
