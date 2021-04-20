import sklearn
import pandas as pd
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from librosa import display

'''
wikimedia:
commons.wikimedia.org/wiki/File:FFT-Time-Frequency-View.png
package doc:
https://librosa.org/doc/main/generated/librosa.load.html
'''

genre = 'pop'
#genre = 'classical'
num = '00000'


audio_file = 'data/genres/'+genre+'/'+genre+'.'+ num +'.wav'
## sr: sample rate
## x : audio time series in the form of numpy.ndarray
x, sr = librosa.load(audio_file, sr = 44100)

print(x)


#----------------#
wave_plot = plt.figure(figsize=(13,5))
display.waveplot(x, sr=sr,alpha = 0.3)
wave_plot.savefig('wave_plot_classical.00000.png')
plt.close()

stft_data = librosa.stft(x)
stft_data_db = librosa.amplitude_to_db(abs(stft_data))

#----------------#
## different color measure the density of the wave
## red stands for there are many sound curve at tha Hz frequency rate
spectrogram = plt.figure(figsize=(13,5))
display.specshow(stft_data_db, sr = sr, x_axis='time', y_axis='hz')
plt.colorbar()
spectrogram.savefig('spectrogram_'+genre+num+'.png')
plt.close()


#----------------#
## we want the weighted average of the Hz, weighted by the density.
spectral_centroids = librosa.feature.spectral_centroid(x, sr = sr)[0]
print(spectral_centroids)

spectral_centroids_plot = plt.figure(figsize=(13,5))
frames = range(len(spectral_centroids))
# time
t = librosa.frames_to_time(frames)

rescale_centroids = sklearn.preprocessing.minmax_scale(spectral_centroids,axis = 0)
plt.plot(t,rescale_centroids, color = 'r')
display.waveplot(x, sr=sr,alpha = 0.3)
spectral_centroids_plot.savefig('spectral_centroids_'+genre+num+'.png')
plt.close()




#----------------#
## rolloff

## rolloff doesn't work if x = 0
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr = sr)[0]
spectral_rolloff_plot = plt.figure(figsize=(13,5))
display.waveplot(x, sr=sr, alpha = 0.3)
rescale_rolloff = sklearn.preprocessing.minmax_scale(spectral_rolloff,axis = 0)
plt.plot(t,rescale_rolloff, color = 'r')
spectral_rolloff_plot.savefig('spectral_rolloff_'+genre+num+'.png')
plt.close()






