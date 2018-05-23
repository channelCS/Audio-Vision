# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 03:37:44 2018

@author: adityac8
"""

import warnings
warnings.simplefilter("ignore")

import sys
ka_path="C:/Users/aditya/version-control"
sys.path.insert(0, ka_path)
from keras_aud import aud_audio, aud_feature
from keras_aud import aud_model

import csv
import cPickle
import numpy as np
import scipy
import time
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from keras.utils import to_categorical
from keras.models import load_model

wav_dev_fd   = ka_path+'/dcase_data/audio/dev'
wav_eva_fd   = ka_path+'/dcase_data/audio/eva'
dev_fd       = ka_path+'/dcase_data/features/dev/logmel'
eva_fd       = ka_path+'/dcase_data/features/eva/logmel'
label_csv    = ka_path+'/dcase_data/texts/dev/meta.txt'
txt_eva_path = ka_path+'/dcase_data/texts/eva/test.txt'
new_p        = ka_path+'/dcase_data/texts/eva/evaluate.txt'

#aud_audio.extract('logmel', wav_dev_fd, dev_fd,'defaults.yaml')
#aud_audio.extract('logmel', wav_eva_fd, eva_fd,'example.yaml')

labels = [ 'bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home', 'beach', 
            'library', 'metro_station', 'office', 'residential_area', 'train', 'tram', 'park' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

prep='dev'               # Which mode to use
folds=4                   # Number of folds
#Parameters that are passed to the model.
model_type='Dynamic'   # Type of model
model='CNN'               # Name of model
feature="logmel"          # Name of feature


epochs=10              # Number of Epochs
batchsize=128          # Batch Size
num_classes=15         # Number of classes
#Parameters that are passed to the features.
agg_num=10             # Agg Number(Integer) Number of frames
hop=10                 # Hop Length(Integer)

paul=aud_model.Feature(feature=feature)

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
        try:
            X = cPickle.load( open( path, 'rb' ) )
        except Exception as e:
            print 'Error while parsing',path
            continue
        # reshape data to (n_block, n_time, n_freq)
        i+=1
        X3d = aud_model.mat_2d_to_3d( X, agg_num, hop )
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
        X0 = aud_model.mat_2d_to_3d( X0, agg_num, hop )
        
        X0 = aud_model.mat_3d_to_nd(model,X0)
    
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

tr_X, tr_y = GetAllData( dev_fd, label_csv, agg_num, hop )

print(tr_X.shape)
print(tr_y.shape)    
    
tr_X=aud_model.mat_3d_to_nd(model,tr_X)
print(tr_X.shape)
dimx=tr_X.shape[-2]
dimy=tr_X.shape[-1]

if prep=='dev':
    cross_validation=True
else:
    cross_validation=False
   
cnn_layers=2
acts          = ['relu','relu']
drops         = [0.1   , 0.1  ]
pools         = [['max',2],['max',2]]
nb_filter     = [10    , 10   ]
filter_length = [3     , 3    ]
end_dense={'input_neurons':200,'activation':'relu','dropout':0.1}

miz=aud_model.Dynamic_Model(model = model, cnn_layers = cnn_layers,
                            nb_filter = nb_filter, filter_length = filter_length,
                            dimx = dimx, dimy = dimy,
                            acts = acts, drops = drops, pools = pools,
                            end_dense = end_dense,num_classes = num_classes)

np.random.seed(68)
if cross_validation:
    kf = KFold(len(tr_X),folds,shuffle=True,random_state=42)
    results=[]    
    for train_indices, test_indices in kf:
        train_x = [tr_X[ii] for ii in train_indices]
        train_y = [tr_y[ii] for ii in train_indices]
        test_x  = [tr_X[ii] for ii in test_indices]
        test_y  = [tr_y[ii] for ii in test_indices]
        train_y = to_categorical(train_y,num_classes=len(labels))
        test_y = to_categorical(test_y,num_classes=len(labels)) 
        
        train_x=np.array(train_x)
        train_y=np.array(train_y)
        test_x=np.array(test_x)
        test_y=np.array(test_y)
        print "Development Mode"

        #get compiled model
        lrmodel=miz.prepare_model()

        if lrmodel is None:
            print "If you have used Dynamic Model, make sure you pass correct parameters"
            raise SystemExit
        #fit the model
        lrmodel.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=1)
        
        #make prediction
        pred=lrmodel.predict(test_x, batch_size=32)

        pred = [ii.argmax()for ii in pred]
        test_y = [ii.argmax()for ii in test_y]

        results.append(accuracy_score(pred,test_y))
        print accuracy_score(pred,test_y)
        jj=str(set(list(test_y)))
        print "Unique in test_y",jj
    print "Results: " + str( np.array(results).mean() )
else:
    train_x=np.array(tr_X)
    train_y=np.array(tr_y)
    print "Evaluation mode"
    lrmodel=miz.prepare_model()
    train_y = to_categorical(train_y,num_classes=len(labels))
        
    #fit the model
    lrmodel.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=1)
    
    truth,pred=test(lrmodel,txt_eva_path,new_p,model)

    acc=aud_model.calculate_accuracy(truth,pred)
    print "Accuracy %.2f prcnt"%acc