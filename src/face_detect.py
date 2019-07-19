import sys
sys.path.append('../')
from src.model import VggFace
from utils import load, verify, preprocess_image
import numpy as np
import pandas as pd

# class
class initialization():
	def __init__(self):
		self.embedding_path = './trained-models/embedding/embedded_vector.txt'
		self.name_path = './trained-models/embedding/name.txt'
		self.weights_path = './trained-models/weights/finetune_50_5.h5'
		self.model = VggFace(self.weights_path, is_origin = False)
		self.database = np.loadtxt(self.embedding_path)
		self.name = load(self.name_path)
		
	
	def predict(self, file):
		#vector = model.predict(preprocess_image(self.file))[0,:]
		#b = np.loadtxt(self.embedding)
		self.file = file
		result = verify(self.file, self.model, self.database, self.name)
		return result
	
	def check(self) :
		img = preprocess_image(self.file)
		return img.shape


# class CelebClassifier():
# 	def __init__(self, img):
# 		self.embedding_path = './database/embedded_vector.txt'
# 		self.name_path = './database/name.txt'
# 		self.weights_path = './weights/finetune_50_5.h5'
# 		self.model = VggFace(self.weights_path, is_origin = False)
# 		self.database = np.loadtxt(self.embedding_path)
# 		self.name = load(self.name_path)
# 		self.file = img

# 	def predict(self):
# 		#vector = model.predict(preprocess_image(self.file))[0,:]
# 		#b = np.loadtxt(self.embedding)
# 		result = verify(self.file, self.model, self.database, self.name)
# 		return result
	
# 	def check(self) :
# 		img = preprocess_image(self.file)
# 		return img.shape

class face_detect(initialization):
	def __init__(self):
		super(face_detect, self).__init__()

	

	
#result = face_detect('C:/Users/GT/Desktop/face_ss/dm_ss.png').predict()	
#print(result)




		
			

""""
# load model
model = VggFace(
    path = 'C:/Users/GT/Desktop/model2/weights/finetune_50_5.h5',
                              is_origin = False)
                              

# check model
#print(model.summary())


#load the database
b = np.loadtxt('C:/Users/GT/Desktop/model2/database/embedded_vector.txt')


    

# check
#verify('C:/Users/GT/Desktop/face_ss/dm_ss.png')        


"""
