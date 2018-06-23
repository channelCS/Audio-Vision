from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD
import cv2, numpy as np
import h5py
import json
from glob import glob
import keras.backend as K
K.set_image_dim_ordering('tf')

### load json 

def get_model(weights_path=None):
    
    ## [17-june-2018]Use residual after this
    input_tensor = Input(shape=(448,448,3))
    base_model = VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
        
    model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)  
    model.summary()
    #model = VGG19(weights_path)
    #model.summary()
    return model


def extract(path):
    im = cv2.imread(path)
    #img = image.load_img(path, target_size=(448,448))
    if im is None:
        raise Exception("Incorrect path")
    #im = cv2.resize(im, (448, 448))
    #im = im.transpose((2,0,1))
    #im = np.expand_dims(im, axis=0)
    im = cv2.resize(im, (448,448)).astype(np.float32)
    im = im * 255
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    #im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    im = preprocess_input(im)
#    print (im.shape)
    # Test pretrained model
    model = get_model()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    
    return out
    
    
    
    
path_to_images='E:/akshita_workspace/Audio-Vision/VIS-LSTM/data/coco' #Give path where COCO images reside
features_path='E:/akshita_workspace/Audio-Vision/VIS-LSTM/data/features'

with open('data/vqa_data_prepro.json') as f:
        data = json.load(f)

type_="train"
mydata=data['unique_img_'+type_]              
feat=[]
for i in mydata:
    img=i.split('/')[-1]
    ftr = extract(path_to_images+'/'+img)
    feat.append(ftr)

print (feat.shape())
bre
h5f_data = h5py.File(features_path + '/'+type_+'.h5', 'w')
h5f_data.create_dataset('images_'+ type_, data=np.array(feat))
h5f_data.close()


