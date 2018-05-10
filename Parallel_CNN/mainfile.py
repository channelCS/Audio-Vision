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
ka_path="e:/akshita_workspace/cc"
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
audio_ftr_path='E:/akshita_workspace/git_x/'

# We now tell the paths for audio, features and texts.
wav_dev_fd   = audio_ftr_path+'dcase_data/audio/dev'
wav_eva_fd   = audio_ftr_path+'dcase_data/audio/eva'
dev_fd       = audio_ftr_path+'dcase_data/features/dev'
eva_fd       = audio_ftr_path+'dcase_data/features/eva'
label_csv    = '../utils/dcase_data/texts/dev/meta.txt'
txt_eva_path = '../utils/dcase_data/texts/eva/test.txt'
new_p        = '../utils/dcase_data/texts/eva/evaluate.txt'

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
feature="cqt"             # Name of feature

dropout1=0.1              # 1st Dropout
act1='relu'               # 1st Activation
act2='relu'               # 2nd Activation
act3='softmax'            # 3rd Activation

input_neurons=400         # Number of Neurons
epochs=2                  # Number of Epochs
batchsize=128             # Batch Size
num_classes=15            # Number of classes
filter_length=3           # Size of Filter
nb_filter=100             # Number of Filters

agg_num=10                # Number of frames
hop=10                    # Hop Length

#We extract audio features
#aud_audio.extract(feature, wav_dev_fd, dev_fd+'/'+feature,'example.yaml')
#aud_audio.extract(feature, wav_eva_fd, eva_fd+'/'+feature,'example.yaml')

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
    
miz=aud_model.Functional_Model(input_neurons=input_neurons,dropout=dropout1,
    act1=act1,act2=act2,act3=act3,nb_filter = nb_filter, filter_length=filter_length,
    num_classes=num_classes,
    model=model,dimx=dimx,dimy=dimy)

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

    acc=aud_utils.calculate_accuracy(truth,pred)
    print "Accuracy %.2f prcnt"%acc