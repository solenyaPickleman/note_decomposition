import pandas as pd
import numpy as np
import cupy as cp
import cusignal
from scipy.io import wavfile
from scipy import signal
import time
# import torch
# import torchaudio 
from matplotlib import pyplot as plt 
import json

#build note lookup table
from math import log2, pow

with open ("/home/sir/audiosignal/nsynth-valid/examples.json", "r") as file:
    data = json.load(file)

midi_pitches = {}
freq_pitches = {}
with open('/home/sir/audiosignal/midi_pitches.tsv','r') as file:
    for line in file:
        line = line.strip()
        line = line[:-1] if line[-1]==',' else line 
        note, freq, index = line.split(',')
        midi_pitches[int(index)]=note 
        if freq == '-': continue
        freq_pitches[float(freq)]=note
freq_index = np.array(list(freq_pitches.keys()))

# A4 = 440
# C0 = A4*pow(2, -4.75)
# name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
# def pitch(freq):
#     h = round(12*log2(freq/C0))
#     octave = h // 12
#     n = h % 12
#     return name[n] + str(octave)
def pitch(freq):
    #find closest freq in freq_index 
    diffs = np.abs(freq_index-freq)
    mindiff = np.min(diffs)
    index = np.where(diffs == mindiff)[0][0]
    return freq_pitches[freq_index[index]]

basepath = "/home/sir/audiosignal/nsynth-valid/audio/"


exfiles = list(data.keys())

percentcorrect=0.0
percentincomponents=0.0
by_mode={'electronic':0, 'acoustic':0, 'synthetic':0}
modetotal= {'electronic':0, 'acoustic':0, 'synthetic':0}
times = []
by_instrument={}
instrumenttotal= {}

sigs = []
for exfile in exfiles: 
    if data[exfile]['pitch'] not in midi_pitches : continue
    start = time.time()
    filepath = basepath+exfile+'.wav'
    sample_rate, sig = wavfile.read(filepath)
    sigs.append(sig)

fft = np.fft.fft(sigs)
fft = np.abs(fft[:, :len(fft[0])//2])
maxval = np.amax(fft[:, 1:],1 )
maxindex = []
for i, arr in enumerate(fft):
    maxindex.append(np.where(arr==maxval[i])[0][0])
maxindex = np.array(maxindex)
hz_of_signal = (sample_rate/(len(fft)*2)) * maxindex

period = len(sigs[0])/maxindex
pitch_vectorized = np.vectorize(pitch)
original_pitch = pitch_vectorized(hz_of_signal)



for exfile in exfiles: 
    if data[exfile]['pitch'] not in midi_pitches : continue
    start = time.time()
    filepath = basepath+exfile+'.wav'
    sample_rate, sig = wavfile.read(filepath)

    # plt.plot(sig[:1000])
    # plt.show()

    fft = np.fft.fft(sig)
    fft = np.abs(fft[:len(fft)//2])

    # plt.plot(fft)
    # plt.show()


    #find max peak -- would the note be the FIRST peak ? akak, beginning of components 
    maxval = np.max(fft[1:])
    maxindex = np.where(fft==maxval)[0][0]
    hz_of_signal = (sample_rate/(len(fft)*2)) * maxindex

    period = len(sig)/maxindex
    original_pitch = pitch(hz_of_signal)


    #for instrument classification via goooood ollllllld fassssshhhhhion nn
    #use top 8 overtones as a vector, scaled by ratio of max

    #to find overtones, find indices that are multiples of BASE max freq

    peaks, _ = signal.find_peaks(fft, distance= period*3)
    maxdistance = 0.05

    def closest_to(x):
        base = np.round(x)
        return (base, np.abs(x-base))


    if  maxindex not in peaks: 
        #print('weirdness in ', exfile)
        continue 


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


    pitches = [pitch((sample_rate/(len(fft)*2)) * peaks[x]) for x in above]
    # print(exfile, original_pitch, midi_pitches[data[exfile]['pitch']], pitches)
    times.append(time.time()-start)
    percentcorrect += int(original_pitch ==midi_pitches[data[exfile]['pitch']])
    percentincomponents += int(midi_pitches[data[exfile]['pitch']] in pitches)
    mode = exfile.split('_')[1]
    by_mode[mode]+= int(original_pitch ==midi_pitches[data[exfile]['pitch']])
    modetotal[mode]+=1
    instrum = exfile.split('_')[0]
    for instrument in [instrum, instrum+mode]:
        if instrument not in by_instrument:
            by_instrument[instrument] = 0
            instrumenttotal[instrument] = 0
        instrumenttotal[instrument] += 1
        by_instrument[instrument] +=  int(original_pitch ==midi_pitches[data[exfile]['pitch']])


print("Percent Correct: ", percentcorrect/len(exfiles))
print("Percent in Components: ", percentincomponents/len(exfiles))
for mode in by_mode:
    print("Percent correct for mode ", mode , " : ", by_mode[mode]/modetotal[mode])
for instrument in sorted(by_instrument.keys()):
    print("Percent correct for instrument ", instrument , " : ", by_instrument[instrument]/instrumenttotal[instrument])
print("This took an average of " ,np.mean(times), " seconds per file")

# components = fft[peaks[above]]
# components = (sample_rate/(len(fft)*2))*components
# #scale by max:
# components = components/np.max(components)

#components is now a scaled vector - for nn similarity, we want to "search" by 
# original note , and then by similarity to these vectors discover 
# first what instrument it is, and then see if this approach can get back to a specific
# instrument


