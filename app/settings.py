import os


if '__file__' in vars():
    wk_dir = os.path.dirname(os.path.realpath('__file__'))
    
MODELS_ROOT = os.path.abspath(wk_dir + "/../celebrities-recognition/trained-models/weights")

DATA_ROOT = os.path.abspath(wk_dir +"/../celebrities-recognition/face")

LOG_FILE = os.path.abspath(wk_dir +"/../celebrities-recognition/logs/application.log")

PORT = 5000
