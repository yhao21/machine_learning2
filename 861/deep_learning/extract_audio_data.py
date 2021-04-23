import librosa
import pandas as pd
import os, glob
import numpy as np

if not os.path.exists('figures'):
    os.mkdir('figures')



df = pd.DataFrame()
genres = [
        'blues',
        'classical',
        'country',
        'disco',
        'hiphop',
        'jazz',
        'metal'
        'pop',
        'reggae',
        'rock',
        ]

for genre in genres:
    print(genre)
    num_of_files = 0
    for filename in glob.glob(os.path.join('data/genres', genre, '*.wav')):
        if num_of_files == 10:
            break
        ## filename : path for the wav file
        print(filename)
        x, sr = librosa.load(filename, sr=44100)
        spectral_centroids = librosa.feature.spectral_centroid(x,sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01,sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(x,sr = sr)
        ## count how many time the wave cross 0. It measure how smooth is the music.
        zero_crossing = librosa.feature.zero_crossing_rate(x)
        chromagram = librosa.feature.chroma_stft(x,sr=sr)
        # detect which part is human sound
        mfcc = librosa.feature.mfcc(x,sr=sr)
        # root mean square
        rms = librosa.feature.rms(x)

        df = df.append({
            "filename":filename,
            "spectral_centroid":np.mean(spectral_centroids),
            'spectral_rolloff':np.mean(spectral_rolloff),
            'spectral_bandwidth':np.mean(bandwidth),
            'zero_crossing':np.mean(zero_crossing),
            'rms':np.mean(rms),
            'mfcc0':np.mean(mfcc[0]),
            'mfcc1':np.mean(mfcc[1]),
            'mfcc2':np.mean(mfcc[2]),
            'mfcc3':np.mean(mfcc[3]),
            'mfcc4':np.mean(mfcc[4]),
            'mfcc5':np.mean(mfcc[5]),
            'mfcc6':np.mean(mfcc[6]),
            'mfcc7':np.mean(mfcc[7]),
            'mfcc8':np.mean(mfcc[8]),
            'mfcc9':np.mean(mfcc[9]),
            'mfcc10':np.mean(mfcc[10]),
            'mfcc11':np.mean(mfcc[11]),
            'mfcc12':np.mean(mfcc[12]),
            'mfcc13':np.mean(mfcc[13]),
            'mfcc14':np.mean(mfcc[14]),
            'mfcc15':np.mean(mfcc[15]),
            'mfcc16':np.mean(mfcc[16]),
            'mfcc17':np.mean(mfcc[17]),
            'mfcc18':np.mean(mfcc[18]),
            'mfcc19':np.mean(mfcc[19]),
            }, ignore_index=True)
        num_of_files += 1

print(df)
df.to_csv('audio_extraction_dataset.csv')













