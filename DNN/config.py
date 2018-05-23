# -*- coding: utf-8 -*-
"""
author :: @adityac8
"""
## SET PATHS ACCORDING TO WHERE DATA SHOULD BE STORED

# This is where all audio files reside and features will be extracted
audio_ftr_path='D:/workspace/aditya_akshita/temp/dcase_data'

# We now tell the paths for audio, features and texts.
wav_dev_fd   = audio_ftr_path+'/audio/dev'
wav_eva_fd   = audio_ftr_path+'/audio/eva'
dev_fd       = audio_ftr_path+'/features/dev'
eva_fd       = audio_ftr_path+'/features/eva'
label_csv    = '../dependencies/keras_aud/utils/dcase16_task1/dev/meta.txt'
txt_eva_path = '../dependencies/keras_aud/utils/dcase16_task1/eva/test.txt'
eva_file     = '../dependencies/keras_aud/utils/dcase16_task1/eva/evaluate.txt'

labels = [ 'bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home', 'beach', 
            'library', 'metro_station', 'office', 'residential_area', 'train', 'tram', 'park' ]
lb_to_id = {lb:id for id, lb in enumerate(labels)}
id_to_lb = {id:lb for id, lb in enumerate(labels)}
