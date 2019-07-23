import sys
sys.path.append('../')
from triplet import *
from data import *
from src.preprocess import *
from src.model import *
from src.pretrain_model import *
from src.face_detect import face_detect 
from utils import *
from detector import detect_faces
from PIL import Image
from align import *
from visualization import show_results

import threading
import zipfile
from werkzeug.utils import secure_filename

root = './../database/'
UPLOAD_FOLDER = './database'
IMG_FOLDER= './face_ss'
global_lock = threading.Lock()
model = face_detect()
#def saveimg(datasetid,images):
#    for i in range(len(images)):
#        img_name = os.path.join(DATASET_BASE,datasetid,images[i].filename)
#        images[i].save(img_name)


#update class in dataset
class UpdateData(threading.Thread):
    def __init__(self, datasetid,class_name, images):
        threading.Thread.__init__(self)
        self.datasetid = datasetid
        self.class_name=class_name
        self.images = images
        
    def run(self):
        list_classes = os.listdir(os.path.join(self.datasetid))
        if self.class_name in list_classes:
            for i in range(len(self.images)):
                img_name = os.path.join(self.datasetid,self.class_name,self.images[i].filename)
                self.images[i].save(img_name)
            return
        else:
            path= os.path.join(self.datasetid,self.class_name)
            os.mkdir(path)
            for i in range(len(self.images)):
                img_name = os.path.join(self.datasetid,self.class_name,self.images[i].filename)
                self.images[i].save(img_name)
            return


#import data set         
class Import_Data (threading.Thread):
    def __init__(self,filename,dataset):
        threading.Thread.__init__(self)
        self.filename = filename
        self.dataset = dataset
    
    def run(self):
        self.savedata(self.filename,self.dataset)
        
    def savedata(self,filename,dataset):
        while global_lock.locked():
            continue

        global_lock.acquire()
        #save file zip
        dataset.save(os.path.join(UPLOAD_FOLDER, filename))
        zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, filename), 'r')
        #extract file zip
        zip_ref.extractall(UPLOAD_FOLDER)
        zip_ref.close()
        global_lock.release()
        
class TrainingThread(threading.Thread):
    def __init__(self,dataset,modelname,batch_size, epochs,lr,types,class_name):
        #threading.Thread.__init__(self)
        super(TrainingThread,self).__init__()
        self.modelname = modelname
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr=lr
        self.types = types
        self.class_name = class_name
    def run(self):
        if self.types == 'train_dataset':
            self.pretrainModel(self.modelname,self.dataset,self.batch_size,self.epochs,self.lr)
        if self.types == 'train_class':
            self.pretrainSingleClass(self.modelname,self.dataset,self.class_name,self.batch_size,self.epochs,self.lr)
    
    def pretrainModel(self,modelname, dataset,batch_size,epochs,lr):
        K.clear_session()
        reader = LFWReader(dir_images=dataset)
        gen_train = TripletGenerator(reader)
        gen_test = TripletGenerator(reader)
        embedding_model, triplet_model = GetModel()
        for layer in embedding_model.layers[-3:]:
            layer.trainable = True
            
        for layer in embedding_model.layers[: -3]:
            layer.trainable = False
        triplet_model.compile(loss=None, optimizer=Adam(lr))
        
        history = triplet_model.fit_generator(gen_train, 
                                  validation_data=gen_test,  
                                  epochs=epochs, 
                                  verbose=1, 
                                    steps_per_epoch=5,
                                  validation_steps=5)
        embedding_model.save_weights('./trained-models/weights/'+modelname+'.h5')
        K.clear_session()
        self.embeddingmodel(modelname)
        
    def pretrainSingleClass(self,modelname,dataset,class_name,batch_size,epochs,lr):
        K.clear_session()
        reader = LFWReader(dir_images=dataset,class_name=class_name)
        gen_train = TripletGeneratorSingleID(reader)
        gen_test = TripletGeneratorSingleID(reader)
        embedding_model, triplet_model = GetModel()
        for layer in embedding_model.layers[-3:]:
            layer.trainable = True
            
        for layer in embedding_model.layers[: -3]:
            layer.trainable = False
        triplet_model.compile(loss=None, optimizer=Adam(lr))
        
        history = triplet_model.fit_generator(gen_train, 
                                  validation_data=gen_test,  
                                  epochs=epochs, 
                                  verbose=1, 
                                    steps_per_epoch=50,
                                  validation_steps=5)
        
        embedding_model.save_weights('./trained-models/weights/'+modelname+'.h5')
        K.clear_session()
        self.embeddingmodel(modelname)
        return

        
      
    def embeddingmodel(self,modelname):
        K.clear_session()
        model = VggFace(
            path = './trained-models/weights/'+modelname+'.h5',
                                      is_origin = False)
        metadata = load_metadata(dataset)
        embedded = np.zeros((metadata.shape[0], 2622 ))
        for i in range(metadata.shape[0]):
            img_emb = model.predict(preprocess_image(metadata[i].image_path()))[0,:]
            embedded[i] = img_emb
       
        # save embedding
        np.savetxt('./trained-models/embedding/embedded_vector.txt', embedded)
        name = []
        for i in range(len(metadata)):
            name.append(metadata[i].name)
        np.savetxt('./trained-models/embedding/name.txt', name,  delimiter=" ", fmt="%s")
        
