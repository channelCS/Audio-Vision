"""
Created on Sat Apr 08 11:48:18 2018

@author: Akshita Gupta
Email - akshitadvlp@gmail.com
"""

import numpy as np
import os
from scipy import signal
from scikits.audiolab import wavread
import librosa
import wavio

def feature_normalize(feature_data):
    """   
    Input:
    Output:
        
    """
    N = feature_data.shape[0]
    S1 = np.sum(feature_data, axis=0)
    S2 = np.sum(feature_data ** 2, axis=0)
    mean=S1/N
    std=np.sqrt((N * S2 - (S1 * S1)) / (N * (N - 1)))
    mean = np.reshape(mean, [1, -1])
    std = np.reshape(std, [1, -1])
    feature_data=((feature_data-mean)/std)
    return feature_data

def convert_mono(wav,mono):
    """   
    Input:
    Output:
        
    """
    if mono and wav.ndim==2:
        return np.mean( wav, axis=-1 )
    else:
        return wav  

def read_audio(library,path,dataset=None):
    """   
    Input: 'str','str','str'
    Output: np.ndarray, int
        
    """
    if dataset is not None:
        library='librosa'
    if library == 'wavread':
        wav,fs,enc = wavread(path)
    elif library == 'librosa' and dataset == 'chime_2016': # chime 2016 with different sampling rate at development
        wav,fs = librosa.load(path,sr=16000.)
    elif library == 'librosa' and dataset == 'dcase_2016': # chime 2016 with different sampling rate at development
        wav,fs = librosa.load(path,sr=44100.)
    elif library =='readwav':
        Struct = wavio.read( path )
        wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
        fs = Struct.rate
    else:
        raise Exception("Dataset not listed")
    return wav, fs
        
#def set_sampling_rate(sr):
    
        
def mel(features,path,dataset=None):
    
    """
    This function extracts mel-spectrogram from audio.
    Make sure, you pass a dictionary containing all attributes
    and a path to audio.
    """
    fsx=features['fs'][0]
    n_mels=features['n_mels'][0]
    #print n_mels
    fmin=features['fmin'][0]
    fmax=features['fmax'][0]
    mono=features['mono'][0]
    hamming_window=features['hamming_window'][0]
    noverlap=features['noverlap'][0]
    detrend=features['detrend'][0]
    return_onesided=features['return_onesided'][0]
    mode=features['mode'][0]
    wav, fs = read_audio('librosa',path,dataset)
    #fsx = librosa.resample(wav,fs, 44100)
    #wav, fs = librosa.load(path)
    wav=convert_mono(wav,mono)
    if fs != fsx:
        raise Exception("Assertion Error. Sampling rate Found {} Expected {}".format(fs,fsx))
    ham_win = np.hamming(hamming_window)
    [f, t, X] = signal.spectral.spectrogram(wav,fs, window=ham_win, nperseg=hamming_window, noverlap=noverlap, detrend=detrend, return_onesided=return_onesided, mode=mode )
    X = X.T

    # define global melW, avoid init melW every time, to speed up.
    if globals().get('melW') is None:
        global melW
        melW = librosa.filters.mel( fs, n_fft=hamming_window, n_mels=n_mels, fmin=fmin, fmax=fmax )
        melW /= np.max(melW, axis=-1)[:,None]
    
    X = np.dot( X, melW.T )
    X = X[:, 0:]
    X=feature_normalize(X)
    return X

def logmel(features,path,library='wavread',dataset=None):
    """
    This function extracts log mel-spectrogram from audio.
    Make sure, you pass a dictionary containing all attributes
    and a path to audio.
    """
    fsx=features['fs'][0]
    n_mels=features['n_mels'][0]
    fmin=features['fmin'][0]
    fmax=features['fmax'][0]
    mono=features['mono'][0]
    hamming_window=features['hamming_window'][0]
    noverlap=features['noverlap'][0]
    detrend=features['detrend'][0]
    return_onesided=features['return_onesided'][0]
    mode=features['mode'][0]
    wav, fs = read_audio(library,path,dataset)
    #print "fs before mono",fs #[DEBUG]
    wav=convert_mono(wav,mono)
    if fs != fsx:
        raise Exception("Assertion Error. Sampling rate Found {} Expected {}".format(fs,fsx))
    ham_win = np.hamming(1024)
    [f, t, X] = signal.spectral.spectrogram(wav,fs, window=ham_win, nperseg=hamming_window, noverlap=noverlap, detrend=detrend, return_onesided=return_onesided, mode=mode )
    X = X.T

    # define global melW, avoid init melW every time, to speed up.
    if globals().get('melW') is None:
        global melW
        melW = librosa.filters.mel( fs, n_fft=hamming_window, n_mels=n_mels, fmin=fmin, fmax=fmax )
        melW /= np.max(melW, axis=-1)[:,None]
        #print "mel"
    
    X = np.dot( X, melW.T )
    X = np.log( X + 1e-8)
    X = X[:, 0:]
    
    X=feature_normalize(X)
    return X

def cqt(features,path,dataset=None):
    """
    This function extracts constant q-transform from audio.
    Make sure, you pass a dictionary containing all attributes
    and a path to audio.
    """    
    fsx = features['fs'][0]
    hop_length = features['hop_length'][0]
    n_bins = features['n_bins'][0]
    bins_per_octave = features['bins_per_octave'][0]
    window = features['window'][0]
    mono=features['mono'][0]
    wav, fs = read_audio('librosa',path,dataset)
    wav=convert_mono(wav,mono)
    if fs != fsx:
        raise Exception("Assertion Error. Sampling rate Found {} Expected {}".format(fs,fsx))
    X=librosa.cqt(y=wav, hop_length=hop_length,sr=fs, n_bins=n_bins, bins_per_octave=bins_per_octave,window=window)
    X=X.T
    X=np.abs(np.log10(X))
    return X


#def mfcc(features,path):
import scipy

def spectralCentroid(features,path):
    #https://gist.github.com/endolith/359724/aa7fcc043776f16f126a0ccd12b599499509c3cc   
    fsx = features['fs'][0]
    mono=features['mono'][0]
    wav, fs, enc = read_audio('wavread',path)
    wav=convert_mono(wav,mono)
    assert fs==fsx
    spectrum = abs(np.fft.rfft(wav))
    normalized_spectrum = spectrum / sum(spectrum)  # like a probability mass function
    normalized_frequencies = np.linspace(0, 1, len(spectrum))
    spectral_centroid = sum(normalized_frequencies * normalized_spectrum)
            
    return spectral_centroid
    
def zcr(features,path):
   fsx = features['fs'][0]
   mono=features['mono'][0]
#   nceps = features['nceps'][0]
   frame_length = features['frame_length'][0]
   hop_length = features['hop_length'][0]
   center = features['center'][0]
   pad = features['pad'][0]
   wav, fs, enc = read_audio('wavread',path)
   wav=convert_mono(wav,mono)
   assert fs==fsx
   X=librosa.feature.zero_crossing_rate(wav, frame_length=frame_length, hop_length=hop_length, center=center, pad=pad)
   X=X.T
   return X

def stft(features,path):
   fsx = features['fs'][0]
   window = features['window'][0]
   mono=features['mono'][0]
#   noverlap=features['noverlap'][0]
#   detrend=features['detrend'][0]
#   return_onesided=features['return_onesided'][0] 
#   nperseg = features['nperseg'][0]
#   nfft = features['nfft'][0]
#   boundary = features['boundary'][0]
#   padded = features['padded'][0]
#   axis = features['axis'][0]
   wav, fs, enc = read_audio('wavread',path)
   wav=convert_mono(wav,mono)
   assert fs==fsx
   ham_win = np.hamming(1024)
   f,t,X = scipy.signal.stft(wav, fs, window=ham_win, nperseg=1024, noverlap=0, nfft=1024, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=0)
   return X
    
#def spectralFlux(spectra, rectify=False):
#    """
#    Compute the spectral flux between consecutive spectra
#    """
#    spectralFlux = []
    
    # Compute flux for zeroth spectrum

#    flux = 0
#    for bin in spectra[0]:
#        flux = flux + abs(bin)
      
#   spectralFlux.append(flux)
    
    # Compute flux for subsequent spectra
#    for s in range(1, len(spectra)):
#        prevSpectrum = spectra[s - 1]
#        spectrum = spectra[s]
        
#        flux = 0
#        for bin in range(0, len(spectrum)):
#            diff = abs(spectrum[bin]) - abs(prevSpectrum[bin])
#            
#            # If rectify is specified, only return positive values
#            if rectify and diff < 0:
#                diff = 0
#            
#            flux = flux + diff
#            
#        spectralFlux.append(flux)
#        
#    return spectralFlux
    

def SpectralRolloff(features,path):
    fsx = features['fs'][0]
    mono=features['mono'][0]
    noverlap=features['noverlap'][0]
    detrend=features['detrend'][0]
    return_onesided=features['return_onesided'][0]
    mode=features['mode'][0]
#    window = features['window'][0]
#    noverlap=features['noverlap'][0]
#    input_onesided=features['input_onesided'][0] 
#    nperseg = features['nperseg'][0]
#    nfft = features['nfft'][0]
#    boundary = features['boundary'][0]
#    hop_length = features['hop_length'][0]
#    roll_percent = features['roll_percent'][0]
#    freq = features['freq'][0]
    wav, fs, enc = read_audio('wavread',path)
    wav=convert_mono(wav,mono)
    print wav.shape
    assert fs==fsx
#    ham_win = np.hamming(1024)
#    stft_matrix = stft(features,path)
#    print stft_matrix.shape
    ham_win = np.hamming(1024)
    [f, t, X] = signal.spectral.spectrogram(wav,fs, window=ham_win, nperseg=1024, noverlap=noverlap, detrend=detrend, return_onesided=return_onesided, mode='psd' )
    print X.shape
    rolloff = librosa.feature.spectral_rolloff(wav, sr=fs, S=X, n_fft=1024, hop_length=512, freq=None, roll_percent=0.95)
    rolloff = rolloff.T
    return rolloff

def istft(features,path):
   fsx = features['fs'][0]
   mono=features['mono'][0]
#   window = features['window'][0]
#   noverlap=features['noverlap'][0]
#   input_onesided=features['input_onesided'][0] 
#   nperseg = features['nperseg'][0]
#   nfft = features['nfft'][0]
#   boundary = features['boundary'][0]
#   time_axis = features['time_axis'][0]
#   freq_axis = features['freq_axis'][0]
#   
   wav, fs, enc = read_audio('wavread',path)
   #wav=convert_mono(wav,mono)
   assert fs==fsx
   stft_matrix = stft(features,path)
   t, X = scipy.signal.istft(stft_matrix, fs, window='hann', nperseg=None, noverlap=None, nfft=None, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2)
   
   return X 

#def neural_feature_extracter():
    
    

# hop length = winow length /4    
# https://github.com/jsawruk/pymir/blob/master/pymir/SpectralFlux.py  

#spectrogram is absolute of stft    
    

