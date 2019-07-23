import os


if '__file__' in vars():
    wk_dir = os.path.dirname(os.path.realpath('__file__'))
    
MODELS_ROOT = os.path.abspath(wk_dir + "/../celebrities-recognition/trained-models/weights")

DATA_ROOT = os.path.abspath(wk_dir +"/../celebrities-recognition/face")

LOG_FILE = os.path.abspath(wk_dir +"/../celebrities-recognition/logs/application.log")

UPLOAD_FOLDER= './database'

ALLOWED_EXTENSIONS = set(['txt', 'gif', 'png', 'jpg', 'jpeg', 'bmp', 'rar', 'zip', '7zip', 'doc', 'docx'])

IMG_FOLDER = './face_ss'

PORT = 5000
