import sys
sys.path.append('../')
from triplet import *
from data import *
from src.preprocess import *
from src.model import *
from src.pretrain_model import *

import threading

root = './face'

def saveimg(datasetid,images):
    for i in range(len(images)):
        img_name = os.path.join(DATASET_BASE,datasetid,images[i].filename)
        images[i].save(img_name)

class ImportData(threading.Thread):
    def __init__(self, datasetid, update=False):
        threading.Thread.__init__(self)
        self.datasetid = datasetid
        self.update = update
        
        
    def run(self):
        if self.update:
            
            list_classes = os.listdir(DATASET_BASE)
            class_idx = list_classes.index(self.datasetid)
            pretrainSingleClass(self.datasetid)
            
