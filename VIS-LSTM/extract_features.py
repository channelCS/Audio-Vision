from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.optimizers import SGD
import cv2, numpy as np

def VGG_16(weights_path=None):
    base_model = VGG16(weights='imagenet', pooling = max)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    return model
    
def extract(path):
    im = cv2.resize(cv2.imread(path), (224, 224))
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = VGG_16('vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    return out
    

path_to_images='' #Give path where COCO images reside


feat=[]
from glob import glob
for img in glob(path_to_images+'/*jpg'):
    print img
    ftr = extract(img)
    feat.append(ftr)
    

print feat
    
