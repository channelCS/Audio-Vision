# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 06:06:23 2018

@author: aditya
This file contains modules to be used in `audio` as well as `feature` submodules
"""
import os
import yaml
import feature_description as F
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def rem_all_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def read_yaml(yaml_file):
    if not os.path.exists(yaml_file) or not os.path.isfile(yaml_file):
        print("No yaml files found!! Try Again.")
        return
    with open(yaml_file, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def call_ftr_one(feature_name,featx,wav_file,library,dataset):
    # Don't put dateset=None on the abv line
    """
    Introduce features here
    """
    if feature_name == "mel":
        X = F.mel(featx,wav_file,dataset)
    elif feature_name == "logmel":
        X = F.logmel(featx,wav_file,library,dataset)
    elif feature_name == "cqt":
        X = F.cqt(featx,wav_file,dataset)
    elif feature_name == "spectralCentroid":
        X = F.spectralCentroid(featx,wav_file)
    elif feature_name == "zcr":
        X = F.zcr(featx,wav_file)
    elif feature_name == "stft":
        X = F.stft(featx,wav_file)
    elif feature_name == "istft":
        X = F.istft(featx,wav_file)
    elif feature_name == "SpectralRolloff":
        X = F.SpectralRolloff(featx,wav_file)
    else:
        X = 1000
    return X

def get_list():
    features_list=['mel','logmel','cqt','spectralCentroid','zcr','spectralcentroid','stft','istft','SpectralRolloff']
    return features_list