from datetime import date
import os
from os import listdir
from os.path import join, exists, isdir
import numpy as np

import sys
sys.path.append('../')
from src.config import *
from triplet import *
from data import *
from src.preprocess import *
from src.model import *
from utils import *

from keras.models import Model
from keras.layers import Lambda, Activation, Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from sklearn.preprocessing import normalize

def embeddingmodel():
    model = VggFace(
        path = '/home/sbikevn360/celebrities-recognition/trained-models/weights/weights_finetune_50_5.h5',
                                  is_origin = False)
    metadata = load_metadata('/home/sbikevn360/celebrities-recognition/face')
    embedded = np.zeros((metadata.shape[0], 2622 ))
    for i in range(metadata.shape[0]):
        img_emb = model.predict(preprocess_image(metadata[i].image_path()))[0,:]
        embedded[i] = img_emb
    K.clear_session()
    # save embedding
    np.savetxt('/home/sbikevn360/celebrities-recognition/trained-models/embedding/embedded_vector.txt', embedded)
    name = []
    for i in range(len(metadata)):
        name.append(metadata[i].name)
    np.savetxt('/home/sbikevn360/celebrities-recognition/trained-models/embedding/name.txt', name,  delimiter=" ", fmt="%s")

def pretrainSingleClass(datasetid):
    K.clear_session()
    root = './face'
    reader = LFWReader(dir_images=root,data_name = datasetid)
    gen_train = TripletGeneratorSingleID(reader)
    gen_test = TripletGeneratorSingleID(reader)
    embedding_model, triplet_model = GetModel()
    for layer in embedding_model.layers[-3:]:
        layer.trainable = True
        
    for layer in embedding_model.layers[: -3]:
        layer.trainable = False
    triplet_model.compile(loss=None, optimizer=Adam(0.001))
    
    history = triplet_model.fit_generator(gen_train, 
                              validation_data=gen_test,  
                              epochs=1, 
                              verbose=1, 
                                steps_per_epoch=50,
                              validation_steps=5)
    
    embedding_model.save_weights('/home/sbikevn360/celebrities-recognition/trained-models/weights/weights_finetune_50_5.h5')
    K.clear_session()
    embeddingmodel()
