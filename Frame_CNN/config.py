# -*- coding: utf-8 -*-
"""
author :: @akshitac8
"""
## SET PATHS ACCORDING TO WHERE DATA SHOULD BE STORED

# This is where all audio files reside and features will be extracted
audio_ftr_path='D:/workspace/aditya_akshita/temp/chime_data'

# We now tell the paths for audio, features and texts.
wav_dev_fd     = audio_ftr_path+'/audio/dev'
wav_eva_fd     = audio_ftr_path+'/audio/eva'
dev_fd         = audio_ftr_path+'/features/dev'
eva_fd         = audio_ftr_path+'/features/eva'
meta_train_csv = '../dependencies/keras_aud/utils/dcase16_task4/meta_csvs/development_chunks_refined.csv'
meta_test_csv  = '../dependencies/keras_aud/utils/dcase16_task4/meta_csvs/evaluation_chunks_refined.csv'
label_csv      = '../dependencies/keras_aud/utils/dcase16_task4/label_csvs'

labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]

lb_to_id = {lb:id for id, lb in enumerate(labels)}
id_to_lb = {id:lb for id, lb in enumerate(labels)}
