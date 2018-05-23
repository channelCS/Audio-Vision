'''
SUMMARY:  importing file
AUTHOR:   adityac8
Created:  2018.03.28
Modified: 2018.03.28
--------------------------------------
'''

from __future__ import print_function
from __future__ import division
import os
import cPickle
import matplotlib.pyplot as plt
from librosa.display import waveplot
from librosa.display import specshow
import modules as M
import feature_description as F
import librosa
import numpy as np
from scikits.audiolab import wavread

def save(feat,out_path):
    try:
        cPickle.dump(feat, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise Exception('Error while saving file {}. Exception Caught {}'.format(out_path,e))

def load(path):
    try:
        x = cPickle.load(open(path, 'rb'))
    except Exception as e:
        raise Exception('Error while loading file {}. Exception Caught {}'.format(path,e))
    return x

def plot_fig(y,save='',x_axis='time', max_points=50000.0, offset=0.0, color='#333333', alpha=1.0, show_filename=True, plot=True):
    plt.figure()
    waveplot(y=y,sr=44100,x_axis=x_axis,max_points=max_points,
             offset=offset,color=color,alpha=alpha)
    plt.show()
    if save != '':
        plt.savefig(save)

def plot_sim(y,save=''):
    plt.figure()
    plt.plot(y)
    plt.show()
    if save != '':
        plt.savefig(save)

def plot_spec(y,fs=44100,save='',spec_type='log', hop_length=512, cmap='magma', show_filename=True, show_colorbar=True, plot=True):
    plt.figure()
    if spec_type in ['linear', 'log']:
        D = librosa.core.amplitude_to_db(np.abs(librosa.stft(y.ravel())),ref=np.max)

    elif spec_type.startswith('cqt'):
        D = librosa.core.amplitude_to_db(librosa.cqt(y.ravel(), sr=fs),ref=np.max)

    if spec_type == 'linear':
        specshow(data=D,sr=fs,y_axis='linear',x_axis='time',
                 hop_length=hop_length,cmap=cmap)

    elif spec_type == 'log':
        specshow(data=D,sr=fs,y_axis='log',x_axis='time',
                 hop_length=hop_length,cmap=cmap)

    elif spec_type == 'cqt_hz' or 'cqt':
        specshow(data=D,sr=fs,y_axis='cqt_hz',x_axis='time',
                 hop_length=hop_length,cmap=cmap)

    elif spec_type == 'cqt_note':
        specshow(data=D,sr=fs,y_axis='cqt_note',x_axis='time',
                 hop_length=hop_length,cmap=cmap)

    if show_colorbar:
        plt.colorbar(format='%+2.0f dB')
    plt.show()
    if save != '':
        plt.savefig(save)

def extract_one(feature_name,wav_file,yaml_file='',library='wavread',dataset=None):
    """
    This function extracts features from audio.

    Parameters
    ----------
        feature_name : str
            Name of feature to be extracted
        wav_file : str
            Path to a single audio file
        yaml_file : str
            Path to yaml file
    """
    if feature_name in M.get_list():
        yaml_load = M.read_yaml(yaml_file)
        try:            
            featx=yaml_load[feature_name]
        except Exception as e:
            print("Make sure you add the {} to the YAML file".format(e))
            raise SystemExit
        x = M.call_ftr_one(feature_name,featx,wav_file,library,dataset)
        print("Something wrong happened" if type(x) == 'int' else "Feature found")
        return x
    else:
        print("Invalid Feature Name")

def get_samp(path):
    """   
    Input: str
    Output: int
        
    """
    try:
#        _, fs = librosa.load(path)
        _, fs, _ = wavread(path)
    except:
        raise Exception("File not found",path)
    return fs
