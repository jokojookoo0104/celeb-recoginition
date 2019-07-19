# utils
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def load(fname):
    ''' Load the file using std open'''
    f = open(fname,'r')

    data = []
    for line in f.readlines():
        data.append(line.replace('\n',''))

    f.close()

    return data

# recognition
def verify(img_path, model, b, metadata):
    embedding = model.predict(preprocess_image(img_path))[0,:]
    distance = np.zeros(len(b))
    
    for i in range(len(b)):
        distance[i] = findCosineSimilarity(embedding, b[i])
    min_distance = distance.min()
    if min_distance <= 1:
    	#return min_distance
    	name = metadata[list(distance).index(min_distance)]
    	return str(name)
        #print("... Veryfing")
        #print("Image belong to class {}".format(metadata[list(distance).index(min_distance)].name))
        #print("They are same person")
        #print("The cosin similarity is {}".format(min_distance))
        #plt.imshow(load_img(img_path, target_size=(224, 224)))
        #print(type(metadata[list(distance).index(min_distance)].name))
        
    else:
        print("... Undefined")


 # img resize
def process_image(img):
    min_side = 512
    size = img.shape
    h, w = size[0], size[1]
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv2.resize(img, (new_w, new_h))


    return resize_img       
                  
            

