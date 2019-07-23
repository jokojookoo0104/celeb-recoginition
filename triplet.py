import numpy as np
import os
from os import listdir
from os.path import join, exists, isdir
from PIL import Image
import matplotlib.pyplot as plt
from abc import ABCMeta,abstractmethod


def _CountFiles(root):
    count = 0
    subfolders = [join(root, subfolder) for subfolder in listdir(root)]
    mask = [isdir(subfolder) for subfolder in subfolders]
    count = len(subfolders) - np.sum(mask)
    subfolders = np.array(subfolders)[mask]

    for subfolder in subfolders:
        count += _CountFiles(subfolder)
    return count
       
class DataSetReader:
    __metaclass__ = ABCMeta

    def __init__(self, dir_images):
        self.root = dir_images
        self.data_size = _CountFiles(self.root)
        
        
    @abstractmethod
    def GetTriplet(self):
        pass
      
    def GetDataSize(self):
        return self.data_size
    
    def GetTripletSingleID(self):
        pass
    
class LFWReader(DataSetReader):
    def __init__(self, dir_images,class_name=''):
        super().__init__(dir_images)
        self.dir_images = dir_images
        self.list_classes = os.listdir(self.dir_images) #list
        if class_name !='':
            self.class_name = class_name
            self.class_idx = self.list_classes.index(self.class_name)
        
        self.not_single = [c for c in self.list_classes if len(listdir(join(self.root, c)))>1] #list
        
        self.list_classes_idx = range(len(self.list_classes))
        
        self.not_single_idx = range(len(self.not_single)) # list
        
        self.weights_not_single = [len(listdir(join(self.root, c))) for c in self.not_single]
        self.weights_not_single = np.array(self.weights_not_single)
        self.weights_not_single = self.weights_not_single / np.sum(self.weights_not_single)
        
        self.weights = [len(listdir(join(self.root, c))) for c in self.list_classes]
        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)
        
        
         
    def GetTriplet(self):
        # positive and anchor classes are selected from folders where have more than two pictures
        idx_class_pos = np.random.choice(self.not_single_idx, 1 ,p=self.weights_not_single)[0]
        class_pos = self.not_single[idx_class_pos] 
        dir_pos = join(self.root, class_pos)
        [fileName_img_anchor, fileName_img_pos]= np.random.choice(listdir(dir_pos), 2, replace=False)
        
        # negative classes are selected from all folders
        while True:
            idx_class_neg = np.random.choice(self.list_classes_idx, 1, p=self.weights)[0]
            if idx_class_neg != idx_class_pos:
                break
        class_neg = self.list_classes[idx_class_neg]
        dir_neg = join(self.root, class_neg)
        fileName_img_neg = np.random.choice(listdir(dir_neg), 1)[0]
        

        path_anchor = join(dir_pos, fileName_img_anchor)
        path_pos = join(dir_pos, fileName_img_pos)
        path_neg = join(dir_neg, fileName_img_neg)

        return path_anchor, path_pos, path_neg
    
    def GetTripletSingleID(self):
        idx_class_pos = self.class_idx
        class_pos = self.not_single[idx_class_pos]
        dir_pos = join(self.root,class_pos)
        [fileName_img_anchor,fileName_img_pos] = np.random.choice(listdir(dir_pos),2,replace=False)
        
        #negative classes are selected from all folders
        
        while True:
            idx_class_neg = np.random.choice(self.list_classes_idx,1,p=self.weights)[0]
            if  idx_class_neg !=idx_class_pos:
                break
        class_neg = self.list_classes[idx_class_neg]
        dir_neg = join(self.root,class_neg)
        fileName_img_neg = np.random.choice(listdir(dir_neg),1)[0]
        
        path_anchor = join(dir_pos,fileName_img_anchor)
        path_pos = join(dir_pos,fileName_img_pos)
        path_neg = join(dir_neg,fileName_img_neg)
        
        return path_anchor,path_pos, path_neg
            
class MixedReader(DataSetReader):
    def __init__(self, list_readers):
        self.readers = list_readers
        
    def GetTriplet(self):
        sizes = np.array([reader.GetDataSize() for reader in self.readers])
        p = sizes/np.sum(sizes)
        idx = np.random.choice(range(len(self.readers)), size=1 ,p=p)[0]
        path_anchor, path_pos, path_neg = self.readers[idx].GetTriplet()
        return path_anchor, path_pos, path_neg
    
    def GetTripletSingleID(self):
        sizes = np.array([reader.GetDataSize() for reader in self.readers])
        p = sizes/np.sum(sizes)
        idx = np.random.choice(range(len(self.readers)), size=1 ,p=p)[0]
        path_anchor, path_pos, path_neg = self.readers[idx].GetTriplet()
        return path_anchor, path_pos, path_neg
        
    def GetDataSize(self):
        return np.sum([reader.GetDataSize() for reader in self.readers])            
        
        
        
        
        
        
        
        
        