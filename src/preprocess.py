from keras.models import Model
from keras.layers import Lambda, Activation, Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from keras import backend as K
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import sys
sys.path.append('../')
from src.config import *

def preprocess_input(batch_image):
    if not isinstance(batch_image, (np.ndarray, np.generic)):
        error_msg = "data must be 4d numpy array, but found {}"
        raise TypeError(error_msg.format(type(batch_image)))
    shape = batch_image.shape
    if len(shape) != 4:
        error_msg = "data must be shape of (batch, 224, 224, 3), but found {}"
        raise ValueError(error_msg.format(shape))
    (batch, size0, size1, channel) = shape
    if size0 != 224 or size1 != 224 or channel != 3:
        error_msg = "data must be shape of (batch, 224, 224, 3), but found {}"
        raise ValueError(error_msg.format(shape))
        
    batch_image = batch_image.astype(np.float32)
    batch_image = batch_image.transpose([0,3,1,2])
    batch_image = batch_image.transpose([0,2,3,1])
    return batch_image

def TripletGenerator(reader):
    while True:
        list_pos = []
        list_anchor = []
        list_neg = []

        for _ in range(batch_size):
            path_anchor, path_pos, path_neg = reader.GetTriplet()
            img_anchor = _Flip(_ReadAndResize(path_anchor))
            img_pos = _Flip(_ReadAndResize(path_pos))
            img_neg = _Flip(_ReadAndResize(path_neg))
            list_pos.append(img_pos)
            list_anchor.append(img_anchor)
            list_neg.append(img_neg)

        A = preprocess_input(np.array(list_anchor))
        P = preprocess_input(np.array(list_pos))
        N = preprocess_input(np.array(list_neg))
        label = None
        
        yield ({'anchor_input': A, 'positive_input': P, 'negative_input': N}, label) 

def TripletGeneratorSingleID(reader):
    while True:
        list_pos =[]
        list_neg=[]
        list_anchor=[]
        
        for _ in range(batch_size):
            path_anchor,path_pos,path_neg = reader.GetTripletSingleID()
            img_anchor = _Flip(_ReadAndResize(path_anchor))
            img_pos = _Flip(_ReadAndResize(path_pos))
            img_neg = _Flip(_ReadAndResize(path_neg))
            list_pos.append(img_pos)
            list_anchor.append(img_anchor)
            list_neg.append(img_neg)
        
        A = preprocess_input(np.array(list_anchor))
        P = preprocess_input(np.array(list_pos))
        N = preprocess_input(np.array(list_neg))
        label = None
        yield ({'anchor_input': A,'positive_input':P,'negative_input': N}, label)
        
def _ShowImg(img):
    plt.figure()
    plt.imshow(img.astype('uint8'))
    plt.show()
    plt.close()

def _ReadAndResize(filepath):
    im = Image.open((filepath)).convert('RGB')
    im = im.resize((224, 224))
    return np.array(im, dtype="float32")

def _Flip(im_array):
    if np.random.uniform(0, 1) > 0.7:
        im_array = np.fliplr(im_array)
    return im_array
    
    
    
def _TestTripletGenerator(reader):  
    gen = TripletGenerator(reader)
    data = next(gen)
    imgs_anchor = data[0]['anchor_input']
    imgs_pos = data[0]['positive_input']
    imgs_neg = data[0]['negative_input']
    print(imgs_anchor.shape)
    print(imgs_pos.shape)
    print(imgs_neg.shape)
    #imgs_anchor = imgs_anchor.transpose([0,2,3,1])
    #imgs_pos = imgs_pos.transpose([0,2,3,1])
    #imgs_neg = imgs_neg.transpose([0,2,3,1])
    
    for idx_img in range(batch_size):
        anchor = imgs_anchor[idx_img]
        pos = imgs_pos[idx_img]
        neg = imgs_neg[idx_img]
        print(anchor.shape)
        print(pos.shape)
        print(neg.shape)
        _ShowImg(anchor)
        _ShowImg(pos)
        _ShowImg(neg)
        break
    
    print('data size is {}'.format(reader.GetDataSize()))