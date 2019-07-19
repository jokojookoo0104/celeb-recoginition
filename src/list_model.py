import sys
sys.path.append('../')
from src.config import *
import os
from os import listdir
from os.path import isfile,join

def list_all_models():
    files = []
    for file in os.listdir(MODELS_BASE):
        if isfile(join(MODELS_BASE,file)):
            files.append(file)

    models = []
    
    #load the models to memory only once, when the app boots
    
    for path_to_model in files:
        model = os.path.basename(path_to_model)
        version_id = os.path.splitext(model)[0]
        model = {
            "model": model,
            "version": version_id}
        models.append(model)
    return models