# -*- coding: utf-8 -*-
"""
Created on Tue May 08 17:13:52 2018
author: @adityac8
"""

import warnings
warnings.simplefilter("ignore")

import sys
ka_path="../dependencies"
sys.path.insert(0, ka_path)
from keras_aud import aud_audio, aud_feature
from keras_aud import aud_model, aud_utils

############# MAKE IMPORTS   #####################
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from keras.utils import to_categorical
from get_data import GetValues

np.random.seed(68)

############# GET MODEL PARAMETERS #################
seth = GetValues()
prep, folds, save_model, model_type, model,feature, dropout, act1, act2,\
act3, input_neurons, epochs, batchsize, num_classes, agg_num, hop, loss,\
nb_filter, filter_length, optimizer,\
dataset=seth.get_parameters(epochs=8,dataset='chime_2016')
import config as cfg

############# EXTRACT FEATURES #####################
extract = False
if extract:
    aud_audio.extract(feature, cfg.wav_dev_fd, cfg.dev_fd+'/'+feature,'parameters.yaml',dataset=dataset)
    aud_audio.extract(feature, cfg.wav_eva_fd, cfg.eva_fd+'/'+feature,'parameters.yaml',dataset=dataset)

############# LOAD DATA ###########################
tr_X, tr_y = seth.get_train_data()
dimx=tr_X.shape[-2]
dimy=tr_X.shape[-1]
tr_X=aud_utils.mat_3d_to_nd(model,tr_X)
miz=aud_model.Functional_Model(input_neurons=input_neurons,dropout=dropout,
    num_classes=num_classes,model=model,dimx=dimx,nb_filter=nb_filter,
    act1=act1,act2=act2,act3=act3,
    filter_length=filter_length,dimy=dimy)

if prep=='dev':
    kf = KFold(len(tr_X),folds,shuffle=True,random_state=42)
    results=[]    
    for train_indices, test_indices in kf:
        train_x = [tr_X[ii] for ii in train_indices]
        train_y = [tr_y[ii] for ii in train_indices]
        test_x  = [tr_X[ii] for ii in test_indices]
        test_y  = [tr_y[ii] for ii in test_indices]
        train_y = to_categorical(train_y,num_classes=len(cfg.labels))
        test_y = to_categorical(test_y,num_classes=len(cfg.labels)) 
        
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
elif prep=='eval':
    train_x=np.array(tr_X)
    train_y=np.array(tr_y)
    print "Evaluation mode"
    lrmodel=miz.prepare_model()
        
    #fit the model
    lrmodel.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=1)
    
    truth,pred=seth.get_test_predictions(lrmodel)

    eer=aud_utils.calculate_eer(truth,pred)
    
    p,r,f=aud_utils.prec_recall_fvalue(pred,truth,0.4,'macro')
    print "EER %.2f"%eer
    print "Precision %.2f"%p
    print "Recall %.2f"%r
    print "F1 score %.2f"%f
else:
    raise Exception('Wrong prep',prep)