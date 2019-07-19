import sys
sys.path.append('../')

from keras.models import Model
from keras.layers import Lambda, Activation, Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K
from keras.layers import Input
from sklearn.preprocessing import normalize
import numpy as np


# model
def VggFace(path = '/home/sbikevn360/celebrities-recognition/trained-models/weights/finetune_50_5.h5', is_origin=False):
    img = Input(shape=(224, 224,3))

    #convolution layers
    conv1_1 = Conv2D(64, (3,3), activation='relu', name='conv1_1',padding='same')(img)
    conv1_2 = Conv2D(64, (3,3), activation='relu', name='conv1_2',padding='same')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3,3), activation='relu', name='conv2_1',padding='same')(pool1)
    conv2_2 = Conv2D(128, (3,3), activation='relu', name='conv2_2',padding='same')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3,3), activation='relu', name='conv3_1',padding='same')(pool2)
    conv3_2 = Conv2D(256, (3,3), activation='relu', name='conv3_2',padding='same')(conv3_1)
    conv3_3 = Conv2D(256, (3,3), activation='relu', name='conv3_3',padding='same')(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3,3), activation='relu', name='conv4_1',padding='same')(pool3)
    conv4_2 = Conv2D(512, (3,3), activation='relu', name='conv4_2',padding='same')(conv4_1)
    conv4_3 = Conv2D(512, (3,3), activation='relu', name='conv4_3',padding='same')(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3,3), activation='relu', name='conv5_1',padding='same')(pool4)
    conv5_2 = Conv2D(512, (3,3), activation='relu', name='conv5_2',padding='same')(conv5_1)
    conv5_3 = Conv2D(512, (3,3), activation='relu', name='conv5_3',padding='same')(conv5_2)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool5')(conv5_3)

    #classification layer of original mat file
    fc6 = Conv2D(4096, (7,7), activation='relu', name='fc6',padding='valid')(pool5)
    fc6_drop = Dropout(0.5)(fc6)
    
    fc7 = Conv2D(4096, (1,1), name='fc7',padding='valid')(fc6_drop)
    #norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(fc7)
    
    fc7_activation = Activation('relu')(fc7)
    fc7_drop = Dropout(0.5)(fc7_activation)
    fc8 = Conv2D(2622, (1,1), activation='relu', name='fc8',padding='valid')(fc7_drop)
    flat = Flatten(name='flat')(fc8)
    
    # 
    norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(flat)
    
    prob = Activation('softmax',name='prob')(flat)

    if is_origin:
      model = Model(inputs = img, outputs = prob)
      model._make_predict_function()
      model.load_weights(path)
      return Model(inputs = img, outputs = norm)
    else:
      model = Model(inputs = img, outputs = norm)
      model._make_predict_function()
      model.load_weights(path)
      return model

    
def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)

    
def triplet_loss_np(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = np.square(anchor - positive)
    negative_distance = np.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = np.sqrt(np.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = np.sqrt(np.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = np.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = np.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = np.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = np.log(1 + np.exp(loss))
    return np.mean(loss)

def check_loss():
    batch_size = 10
    shape = (batch_size, 4096)

    p1 = normalize(np.random.random(shape))
    n = normalize(np.random.random(shape))
    p2 = normalize(np.random.random(shape))
    
    input_tensor = [K.variable(p1), K.variable(n), K.variable(p2)]
    out1 = K.eval(triplet_loss(input_tensor))
    input_np = [p1, n, p2]
    out2 = triplet_loss_np(input_np)

    assert out1.shape == out2.shape
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1-out2))

    
def GetModel():
    # get the embedded of the image
    embedding_model = VggFace(path = '/home/sbikevn360/celebrities-recognition/trained-models/weights/finetune_50_5.h5', is_origin=False)
    # set the standard size of image input for vgg model
    input_shape = (224, 224, 3)
    # set the input for the model
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')
    # calculate the embedding for each anchor, positive and negative
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)
    # produce the output in the form of array
    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]
       
    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))
    return embedding_model, triplet_model