# rVADfast
The Python library for an unsupervised, fast method for robust voice activity detection (rVAD), as presented in [rVAD: An Unsupervised Segment-Based Robust Voice Activity Detection Method, Computer Speech & Language, 2020](https://www.sciencedirect.com/science/article/pii/S0885230819300920) or its [arXiv version](https://arxiv.org/abs/1906.03588). 
More info on [the rVAD GitHub page](https://github.com/zhenghuatan/rVAD). 

***The rVAD paper published in Computer Speech & Language won International Speech Communication Association (ISCA) 2022 Best Research Paper Award.***

The rVAD method consists of two passes of denoising followed by a VAD stage. It has been applied as a preprocessor for 
a wide range of applications, such as speech recognition, speaker identification, language identification, age and 
gender identification, self-supervised learning, human-robot interaction, audio archive segmentation, 
and so on as in [Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=fugL2E8AAAAJ&citation_for_view=fugL2E8AAAAJ:-mN3Mh-tlDkC).  

The method is unsupervised to make it applicable to a broad range of acoustic environments, 
and it is optimized considering both noisy and clean conditions. 

The rVAD (out of the box) ranks the 4th place (out of 27 supervised/unsupervised systems) 
in a Fearless Steps Speech Activity Detection Challenge. 

The rVAD paper is among [the most cited articles from Computer Speech and Language published since 2018](https://www.journals.elsevier.com/computer-speech-and-language/most-cited-articles) (the 6th place), in 2023.

## Usage
The [rVADfast](https://pypi.org/project/rVADfast/) library is available as a python package installable via: 
```bash
pip install rVADfast
```
After installation, you can import the rVADfast class 
from which you can instantiate a VAD instance which you can use to generate vad labels:
```python
import audiofile
from rVADfast import rVADfast

vad = rVADfast()

path_to_audiofile = "some_audio_file.wav"

waveform, sampling_rate = audiofile.read(path_to_audiofile)
vad_labels, vad_timestamps = vad(waveform, sampling_rate)

```

The package also contains functionality to process folders of audio files, to generate VAD labels 
or to trim non-speeh segments from audio files.
This is done by importing the ```rVADfast.process``` module which has two methods for processing audio files, 
namely ```process.rVADfast_single_process``` and ```process.rVADfast_multi_process```, 
with the latter utilizing multiple CPUs for processing.
Additionally, a processing script can be called from commandline-tools by executing: 
```bash
rVADfast_process --root <audio_file_root> --save_folder <path_to_save_files> 
--ext <audio_file_extension> --n_workers <number_of_multiprocessing_workers>
```
For an explanation of the additional available arguments for the commandline tool you can use: 
```bash
rVADfast_process --help
```

In ```/notebooks``` a concrete example on how to use the rVADfast package is found.

*Note that the package is still in development.
Therefore, we welcome any feedback or suggestions for changes and/or additional features.*

## References
1) Z.-H. Tan, A.k. Sarkara and N. Dehak, "rVAD: an unsupervised segment-based robust voice activity detection method," Computer Speech and Language, vol. 59, pp. 1-21, 2020. 
2) Z.-H. Tan and B. Lindberg, "Low-complexity variable frame rate analysis for speech recognition and voice activity detection,” IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.
