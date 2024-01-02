# rVADfast
The Python library for an unsupervised, fast method for robust voice activity detection (rVAD), as presented in the paper rVAD: An Unsupervised Segment-Based Robust Voice Activity Detection Method, Computer Speech and Language, 2020. More info on [the rVAD GitHub page](https://github.com/zhenghuatan/rVAD). 

The rVAD method consists of two passes of denoising followed by a VAD stage. It has been applied as a preprocessor for a wide range of applications, such as speech recognition, speaker identification, language identification, age and gender identification, self-supervised learning, human-robot interaction, audio archive segmentation, and so on as in [Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=fugL2E8AAAAJ&citation_for_view=fugL2E8AAAAJ:-mN3Mh-tlDkC).  

The method is unsupervised to make it applicable to a broad range of acoustic environments, and it is optimized considering both noisy and clean conditions. 

The rVAD (out of the box) ranks the 4th place (out of 27 supervised/unsupervised systems) in a Fearless Steps Speech Activity Detection Challenge. 

The rVAD paper is among [the most cited articles from Computer Speech and Language published since 2018](https://www.journals.elsevier.com/computer-speech-and-language/most-cited-articles) (the 6th place), in 2023.
