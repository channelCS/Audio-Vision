# -*- coding: utf-8 -*-
"""
Created on Tue May 08 19:06:33 2018

@author: adityac8
"""

# Suppress warnings
import warnings
warnings.simplefilter("ignore")

# Clone the keras_aud library and place the path in ka_path variable
import sys
ka_path="e:/akshita_workspace/git_x"
sys.path.insert(0, ka_path)
from keras_aud import aud_feature

dcase_dev = 'E:/akshita_workspace/git_x/dcase_data/audio/dev/a001_0_30.wav'
chime_dev = 'E:/akshita_workspace/git_x/chime_data/audio/dev/CR_lounge_200110_1601.s300_chunk10.wav'
chime_eva = 'E:/akshita_workspace/git_x/chime_data/audio/eva/CR_lounge_200110_1601.s0_chunk0.wav'
fs1 = aud_feature.get_samp(dcase_dev)
fs2 = aud_feature.get_samp(chime_dev)
fs3 = aud_feature.get_samp(chime_eva)
print fs1
print fs2
print fs3
X1 = aud_feature.extract_one(feature_name = 'logmel', wav_file = dcase_dev, yaml_file = 'yaml/dcase.yaml', dataset='dcase_2016')
X2 = aud_feature.extract_one(feature_name = 'logmel', wav_file = chime_dev, yaml_file = 'yaml/chime.yaml', dataset = 'chime_2016')
X3 = aud_feature.extract_one(feature_name = 'logmel', wav_file = chime_eva, yaml_file = 'yaml/chime.yaml', dataset = 'chime_2016')
aud_feature.plot_spec(X1,fs1)
#aud_feature.plot_spec(X2,fs2)
#aud_feature.plot_spec(X3,fs3)
#print X
#aud_feature.plot_fig(X)
#aud_feature.plot_sim(X)