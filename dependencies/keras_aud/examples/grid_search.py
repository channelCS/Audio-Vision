# -*- coding: utf-8 -*-
"""
Created on Tue May 08 17:13:52 2018

@author: adityac8
"""

# Suppress warnings
import warnings
warnings.simplefilter("ignore")

# Clone the keras_aud library and place the path in ka_path variable
import sys
ka_path="e:/akshita_workspace/git_x"
sys.path.insert(0, ka_path)
from keras_aud import aud_audio, aud_model, aud_utils

# Make imports
import csv
import cPickle
import numpy as np
import scipy
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from keras.utils import to_categorical

# This is where all audio files reside and features will be extracted
audio_ftr_path='E:/akshita_workspace/git_x'
util_path='E:/akshita_workspace/git_x/Summaries'
# We now tell the paths for audio, features and texts.
wav_dev_fd   = audio_ftr_path+'/dcase_data/audio/dev'
wav_eva_fd   = audio_ftr_path+'/dcase_data/audio/eva'
dev_fd       = audio_ftr_path+'/dcase_data/features/dev'
eva_fd       = audio_ftr_path+'/dcase_data/features/eva'
label_csv    = util_path+'/utils/dcase_data/texts/dev/meta.txt'
txt_eva_path = util_path+'/utils/dcase_data/texts/eva/test.txt'
new_p        = util_path+'/utils/dcase_data/texts/eva/evaluate.txt'

labels = [ 'bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home', 'beach', 
            'library', 'metro_station', 'office', 'residential_area', 'train', 'tram', 'park' ]
lb_to_id = {lb:id for id, lb in enumerate(labels)}
id_to_lb = {id:lb for id, lb in enumerate(labels)}

# We define all model parameters here.
prep='eval'               # dev or eval
folds=4                   # Number of folds
save_model=False          # True if we want to save model
model_type='Functional'   # Can be Dynamic or Functional
model='DNN'               # Name of model
feature="logmel"          # Name of feature

dropout1=0.2              # 1st Dropout
act1='relu'               # 1st Activation
act2='relu'               # 2nd Activation
act3='sigmoid'            # 3rd Activation

input_neurons=200         # Number of Neurons
epochs=100                 # Number of Epochs
batchsize=100              # Batch Size
num_classes=15            # Number of classes

agg_num=10                # Number of frames
hop=10                    # Hop Length

#We extract audio features
#aud_audio.extract(feature, wav_dev_fd, dev_fd+'/'+feature,'example.yaml',dataset = 'dcase_2016')
#aud_audio.extract(feature, wav_eva_fd, eva_fd+'/'+feature,'example.yaml',dataset = 'dcase_2016')  

def GetAllData(fe_fd, csv_file):
    """
    Loads all the features saved as pickle files.
    Input: Features folder(str), CSV file(str)
    Output: Loaded features(np array) and labels(np array).
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
        try:
            X = cPickle.load( open( path, 'rb' ) )
        except Exception as e:
            print 'Error while parsing',path
            continue
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

def test(md,csv_file):
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
        fe_path = eva_fd + '/' + feature + '/' + na + '.f'
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

tr_X, tr_y = GetAllData( dev_fd+'/'+feature, label_csv)

print(tr_X.shape)
print(tr_y.shape)    
    
dimx=tr_X.shape[-2]
dimy=tr_X.shape[-1]
tr_X=aud_utils.mat_3d_to_nd(model,tr_X)
print(tr_X.shape)

if prep=='dev':
    cross_validation=True
else:
    cross_validation=False

np.random.seed(68)

train_x=np.array(tr_X)
train_y=np.array(tr_y)
train_y = to_categorical(train_y,num_classes=len(labels))

for d in [0.1,0.2]:
    for i in [100,200]:
        miz=aud_model.Functional_Model(input_neurons=i,dropout=d,num_classes=num_classes,model=model,dimx=dimx,dimy=dimy)
        lrmodel=miz.prepare_model()
        lrmodel.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=0)
        truth,pred=test(lrmodel,txt_eva_path)
        acc=aud_utils.calculate_accuracy(truth,pred)
        print "Accuracy %.2f prcnt"%acc

