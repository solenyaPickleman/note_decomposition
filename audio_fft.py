import pandas as pd
import numpy as np
import cupy as cp
import cusignal
from scipy.io import wavfile

# import torch
# import torchaudio 
from matplotlib import pyplot as plt 

#build note lookup table
from math import log2, pow

A4 = 440
C0 = A4*pow(2, -4.75)
name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
def pitch(freq):
    h = round(12*log2(freq/C0))
    octave = h // 12
    n = h % 12
    return name[n] + str(octave)


basepath = "/home/sir/audiosignal/nsynth-valid/audio/"
exfile = "guitar_acoustic_014-061-025.wav"

filepath = basepath+exfile
sample_rate, sig = wavfile.read(filepath)

plt.plot(sig[:1000])
plt.show()

fft = np.fft.fft(sig)
fft = np.abs(fft[:len(fft)//2])

plt.plot(fft)
plt.show()


#find first peak
maxval = np.max(fft)
maxindex = np.where(fft==maxval)[0][0]
hz_of_signal = (sample_rate/(len(fft)*2)) * maxindex

period = len(sig)/maxindex
original_pitch = pitch(hz_of_signal)


#for instrument classification via goooood ollllllld fassssshhhhhion nn
#use top 8 overtones as a vector, scaled by ratio of max

#to find overtones, find indices that are multiples of BASE max freq
from scipy import signal

peaks, _ = signal.find_peaks(fft, distance= period*3)
maxdistance = 0.05

def closest_to(x):
    base = np.round(x)
    return (base, np.abs(x-base))


assert maxindex in peaks


below_index = maxindex/peaks 
below = {i:1 for i in range(1,9)}
closest_values = closest_to(below_index)
for multiple, value  in zip(closest_values[0], closest_values[1]):
    if multiple <= 0 or multiple > 8: continue 
    if value > maxdistance: continue
    below[multiple] = min(value, below[multiple])

ks, vs = list(below.keys()), list(below.values())
if 1 in vs:
    ks = ks[:vs.index(1)]
    vs = vs[:vs.index(1)]
below = {k:v for k,v in zip(ks,vs)}

below = [k+v if k+v in below_index else k-v for k,v in below.items()]
below= [np.where(below_index ==x)[0][0] for x in below ]

maxindex = peaks[min(below)]

above_index = peaks/maxindex
above = {i:1 for i in range(1,9)}
closest_values = closest_to(above_index)
for multiple, value  in zip(closest_values[0], closest_values[1]):
    if multiple <= 0 or multiple > 8: continue 
    if value > maxdistance: continue
    above[multiple] = min(value, above[multiple])

ks, vs = list(above.keys()), list(above.values())
if 1 in vs:
    ks = ks[:vs.index(1)]
    vs = vs[:vs.index(1)]
above = {k:v for k,v in zip(ks,vs)}

above = [k+v  if k+v in above_index else k-v for k,v in above.items()]
above= [np.where(above_index ==x)[0][0] for x in above ]

for x in above:
    print(pitch((sample_rate/(len(fft)*2)) * peaks[x]))