# -*- coding: utf-8 -*-
"""
Created on Tue May 08 13:48:37 2018

@author: @akshitac8
"""

# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter("ignore")

import sys
ka_path="../.."
sys.path.insert(0, ka_path)
from keras_aud import aud_audio, aud_feature
from keras_aud import aud_model, aud_utils

import csv
import cPickle
import numpy as np
import scipy
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from keras.utils import to_categorical

## SET PATHS ACCORDING TO WHERE DATA SHOULD BE STORED
 
# This is where all audio files reside and features will be extracted
audio_ftr_path='D:/workspace/aditya_akshita/temp'

# We now tell the paths for audio, features and texts.
wav_dev_fd   = audio_ftr_path+'/dcase_data/audio/dev'
wav_eva_fd   = audio_ftr_path+'/dcase_data/audio/eva'
dev_fd       = audio_ftr_path+'/dcase_data/features/dev/logmel'
eva_fd       = audio_ftr_path+'/dcase_data/features/eva/logmel'
label_csv    = ka_path+'/keras_aud/utils/dcase16_task1/dev/meta.txt'
txt_eva_path = ka_path+'/keras_aud/utils/dcase16_task1/eva/test.txt'
new_p        = ka_path+'/keras_aud/utils/dcase16_task1/eva/evaluate.txt'

labels = [ 'bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home', 'beach', 
            'library', 'metro_station', 'office', 'residential_area', 'train', 'tram', 'park' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }



prep='eval'               # Which mode to use
folds=4                   # Number of folds
#Parameters that are passed to the model.
model_type='Functional'   # Type of model
model='seq2seq'               # Name of model
feature="logmel"          # Name of feature

dropout1=0.1             # 1st Dropout
act1='relu'              # 1st Activation
act2='sigmoid'              # 2nd Activation
act3='softmax'           # 3rd Activation

input_neurons=400      # Number of Neurons
epochs=10              # Number of Epochs
batchsize=128          # Batch Size
num_classes=15         # Number of classes
filter_length=3        # Size of Filter
nb_filter=100          # Number of Filters
#Parameters that are passed to the features.
agg_num=10             # Agg Number(Integer) Number of frames
hop=10                 # Hop Length(Integer)

dataset = 'dcase_2016'
extract = 0

## EXTRACT FEATURES
if extract:
    aud_audio.extract(feature, wav_dev_fd, dev_fd+'/'+feature,'example.yaml',dataset=dataset)
    aud_audio.extract(feature, wav_eva_fd, eva_fd+'/'+feature,'example.yaml',dataset=dataset)

def GetAllData(fe_fd, csv_file, agg_num, hop):
    """
    Input: Features folder(String), CSV file(String), agg_num(Integer), hop(Integer).
    Output: Loaded features(Numpy Array) and labels(Numpy Array).
    Loads all the features saved as pickle files.
    """
    # read csv
    with open( csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    # init list
    X3d_all = []
    y_all = []
    i=0
    for li in lis:
        # load data
        [na, lb] = li[0].split('\t')
        na = na.split('/')[1][0:-4]
        path = fe_fd + '/' + na + '.f'
        X = aud_feature.load(path)
        # reshape data to (n_block, n_time, n_freq)
        i+=1
        X3d = aud_utils.mat_2d_to_3d( X, agg_num, hop )
        X3d_all.append( X3d )
        y_all += [ lb_to_id[lb] ] * len( X3d )
    
    print "Features loaded",i                
    print 'All files loaded successfully'
    # concatenate list to array
    X3d_all = np.concatenate( X3d_all )
    y_all = np.array( y_all )
    
    return X3d_all, y_all



def test(md,csv_file,new_p,model):
    # load name of wavs to be classified
    with open( csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    # do classification for each file
    names = []
    pred_lbs = []
    
    for li in lis:
        names.append( li[0] )
        na = li[0][6:-4]
        #audio evaluation name
        fe_path = eva_fd + '/' + na + '.f'
        X0 = cPickle.load( open( fe_path, 'rb' ) )
        X0 = aud_utils.mat_2d_to_3d( X0, agg_num, hop )
        
        X0 = aud_utils.mat_3d_to_nd(model,X0)
    
        # predict
        p_y_preds = md.predict(X0)        # probability, size: (n_block,label)
        preds = np.argmax( p_y_preds, axis=-1 )     # size: (n_block)
        b = scipy.stats.mode(preds)
        pred = int( b[0] )
        pred_lbs.append( id_to_lb[ pred ] )
    
    pred = []    
    # write out result
    for i1 in xrange( len( names ) ):
        fname = names[i1] + '\t' + pred_lbs[i1] + '\n' 
        pred.append(fname)
        
    print 'write out finished!'
    truth = open(new_p,'r').readlines()
    pred = [i.split('\t')[1].split('\n')[0]for i in pred]
    truth = [i.split('\t')[1]for i in truth]
    pred.sort()
    truth.sort()
    return truth,pred

#def predict_sequence(encoder, decoder, source, n_steps, cardinality):
#    state = encoder.predict(source)
#    target_seq = np.array([0.0 for _ in range(n_steps*cardinality)]).reshape(1, n_steps,cardinality)
#    print target_seq.shape
#    output = list()
#    for t in range(n_steps):
#        y, h, c = decoder.predict([target_seq] + state)
#        output.append(y[0,0,:])
#        state = [h, c]
#        target_seq = y
#    return np.array(output)


tr_X, tr_y = GetAllData( dev_fd, label_csv, agg_num, hop )

print(tr_X.shape)
print(tr_y.shape)    
dimx=tr_X.shape[-2]
dimy=tr_X.shape[-1]
batch_num=tr_X.shape[0]

tr_X=aud_utils.mat_3d_to_nd(model,tr_X)

print(tr_X.shape)
#target_data=np.zeros((batch_num,dimx,dimy))

if prep=='dev':
    cross_validation=True
else:
    cross_validation=False
    
miz=aud_model.Functional_Model(input_neurons=input_neurons,cross_validation=cross_validation,dropout1=dropout1,
    act1=act1,act2=act2,act3=act3,nb_filter = nb_filter, filter_length=filter_length,
    num_classes=num_classes,
    model=model,dimx=dimx,dimy=dimy)

np.random.seed(68)

truth = open(new_p,'r').readlines()
truth = [i.split('\t')[1]for i in truth]
truth.sort()
train_x=np.array(tr_X)
#target_data=train_x[::-1, :, ::-1]
train_y=np.array(tr_y)
train_y = to_categorical(train_y,num_classes=len(labels))
lrmodel=miz.prepare_model()
#fit the model
lrmodel.fit(x=[train_x,train_x],y=train_x,batch_size=batchsize,epochs=1,verbose=1)
bre
#truth,pred=test(lrmodel,txt_eva_path,new_p,model)
target = predict_sequence(encoder, decoder, train_x, dimx, dimy)
print target.shape
bre
acc=aud_utils.calculate_accuracy(truth,pred)
print "Accuracy %.2f prcnt"%acc

