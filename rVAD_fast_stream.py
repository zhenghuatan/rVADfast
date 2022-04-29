import os
import sys
import code
import pyaudio
import wave


CHUNK = 1024 ##number of frames in buffer
FORMAT = pyaudio.paInt16
CHANNELS = 1 #each frame contents 1 sample of audio --> chunk -> 1024 samples in buffer
RATE = 44100 #no of samples per Seconds
dur = 5 # the duration of recording audio chunck (seconds)

def record():

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print(".......................................")
    print("Start recording of 5s chunk of audio")
    print("Stop recording - press clt+c")
    print(".......................................\n")
    frames = []
    cont=0

    try:
         while True:
             data = stream.read(CHUNK)
             frames.append(data)
             cont = cont + CHUNK
             if cont >= dur*RATE:  #for dur seconds audio
                print('Recorded %d seconds audio' %(dur))
                print('rVAD going ...')
                break;

    except KeyboardInterrupt:
           print("Done recording: stored --> output.wav")
    except Exception as e:
           print(str(e))

    sample_width = p.get_sample_size(FORMAT)

    stream.stop_stream()
    stream.close()
    p.terminate()

    return sample_width, frames


def record_to_file(file_path):
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)

    sample_width, frames = record()

    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


if __name__ == '__main__':
    part=0
    while True:
       audPart = 'output'+ str(part)  
       record_to_file(audPart + '.wav')
       cmd = 'python3' + " " + 'rVAD_fast.py' + " " + audPart +'.wav' + " " + audPart+'.txt'
       os.system(cmd)
       print('Result for audio chunk%d written' %(part))
       part = part + 1
