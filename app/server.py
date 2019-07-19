from flask import Flask, request, send_file, render_template,jsonify,json
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import urllib.request
import io
import requests
import threading
import datetime
import numpy as np
from monty.serialization import loadfn
from monty.json import jsanitize

import logging
import pickle
import re
import traceback

import sys
sys.path.append('../')
from app.utils import files
from app import settings

from utils import *
from detector import detect_faces
from PIL import Image
from visualization import show_results
from align import *
from src.pretrain_model import *
from src.thread_utils import *
from src.config import *
from src.model import VggFace
from src.list_model import *

from keras import backend as K 
from src.face_detect import face_detect 

UPLOAD_FOLDER = './face_ss'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__)
clf = None
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = face_detect()

'''Test'''
@app.route('/',methods=['POST','GET'])
def test():
    return 'Kyanon Computer Vision'


'''check-model'''
@app.route('/<version>/check', methods=['GET'])
def check_model(version):
    if models.get(version,None) is not None:
        return ("OK",200)
    else:
        return ("Not OK",200)
    
    
'''List-models'''
@app.route('/list-model',methods=['POST','GET'])
def list_model():
    data = {'success':False}
    if request.method == 'POST' or request.method == 'GET':
        model_ids = list_all_models()
        data['success'] = True
        data['model_ids'] = model_ids
    return jsonify(data)


@app.route('/<version>/predict',methods=['POST'])
def predict(version):
    data = {'success':False}
    if request.method == 'POST':
        if request.files.get('image'):
            f = request.files.get('image')
            type = secure_filename(f.filename).split('.')[1]
            if type not in ALLOWED_EXTENSIONS:
                return 'Invalid type of file'
            if f :
                filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename) )
                f.save(filename)

                
            
        elif request.form['url']:
            try:
                url = request.form.get('url')
                print(url)
                f = urllib.request.urlopen(url)
                filename = url.split('/')[-1]
                filename = secure_filename(filename)
                
                if filename:
                    filename=os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    f.save(filename)
            except:
                    print('Cannot read image from url')
        if filename:
            fn = secure_filename(filename)[:-4]
            min_side = 512
            img = cv2.imread(filename)
            size = img.shape
            h, w  = size[0], size[1]
            if max(w, h) > min_side:
                img_pad = process_image(img)
            else:
                img_pad = img
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],f'{fn}_resize.png'), img_pad)
        
            img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], f'{fn}_resize.png' ))
            bounding_boxes, landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
            pic_face_detect = show_results(img, bounding_boxes, landmarks) # visualize the results
            pic_face_detect.save(os.path.join(app.config['UPLOAD_FOLDER'], f'{fn}_landmark.png' ) )
            crop_size = 224
            scale = crop_size / 112
            reference = get_reference_facial_points(default_square = True) * scale
            for i in range(len(landmarks)):
                facial5points = [[landmarks[i][j], landmarks[i][j + 5]] for j in range(5)]
                warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
                img_warped = Image.fromarray(warped_face)   
                pic_face_crop = img_warped.save(os.path.join(UPLOAD_FOLDER, f'{fn}_{i}_crop.png' ) )

            # face recognition 
            cleb_name = []
            for i in range(len(landmarks)):
                name = models[version].predict(os.path.join(app.config['UPLOAD_FOLDER'], f'{fn}_{i}_crop.png'))
                cleb_name.append(name)
            
            employeeList = []
            for i in range(len(landmarks)):
                for j in bounding_boxes:
                    face = {
                        "bounding_boxes": {
                            "top": j[0],
                            "right": j[1],
                            "left": j[2],
                            "bottom": j[3]},
                        "landmark": landmarks[i],
                        "prediction": cleb_name[i],
                        "success": True}
                employeeList.append(face)
            

                
                
                
    return jsonify(jsanitize(employeeList))
'''Predict'''
@app.route('/predict',methods=['POST'])
def predict_image():
    data = {'success':False}

    if request.method == 'POST':
        if request.files.get('image'):
            f = request.files.get('image')
            type = secure_filename(f.filename).split('.')[1]
            if type not in ALLOWED_EXTENSIONS:
                return 'Invalid type of file'
            if f :
                filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename) )
                f.save(filename)

                
            
        elif request.form['url']:
            try:
                url = request.form.get('url')
                print(url)
                f = urllib.request.urlopen(url)
                filename = url.split('/')[-1]
                filename = secure_filename(filename)
                
                if filename:
                    filename=os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    f.save(filename)
            except:
                    print('Cannot read image from url')
        if filename:
            fn = secure_filename(filename)[:-4]
            min_side = 512
            img = cv2.imread(filename)
            size = img.shape
            h, w  = size[0], size[1]
            if max(w, h) > min_side:
                img_pad = process_image(img)
            else:
                img_pad = img
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],f'{fn}_resize.png'), img_pad)
        
            img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], f'{fn}_resize.png' ))
            bounding_boxes, landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
            pic_face_detect = show_results(img, bounding_boxes, landmarks) # visualize the results
            pic_face_detect.save(os.path.join(app.config['UPLOAD_FOLDER'], f'{fn}_landmark.png' ) )
            crop_size = 224
            scale = crop_size / 112
            reference = get_reference_facial_points(default_square = True) * scale
            for i in range(len(landmarks)):
                facial5points = [[landmarks[i][j], landmarks[i][j + 5]] for j in range(5)]
                warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
                img_warped = Image.fromarray(warped_face)   
                pic_face_crop = img_warped.save(os.path.join(UPLOAD_FOLDER, f'{fn}_{i}_crop.png' ) )

            # face recognition 
            cleb_name = []
            for i in range(len(landmarks)):
                name = model.predict(os.path.join(app.config['UPLOAD_FOLDER'], f'{fn}_{i}_crop.png'))
                cleb_name.append(name)
            
            employeeList = []
            for i in range(len(landmarks)):
                for j in bounding_boxes:
                    face = {
                        "bounding_boxes": {
                            "top": j[0],
                            "right": j[1],
                            "left": j[2],
                            "bottom": j[3]},
                        "landmark": landmarks[i],
                        "prediction": cleb_name[i],
                        "success": True}
                employeeList.append(face)
            

                
                
                
    return jsonify(jsanitize(employeeList))

@app.route('/update-data',methods=['POST'])
def update_data():
    data = {'success':False}
    if request.method == 'POST':
        if request.form['datasetname']:
            try:
                datasetid =  request.form.get('datasetname')
                file_name=[]
                images = request.files.getlist('image')
                for i in range(len(images)):     #image will be the key 
                    img_name = images[i].filename
                    file_name.append(img_name)
                for i in range(len(images)):
                    img_name = os.path.join(DATASET_BASE,datasetid,images[i].filename)
                    images[i].save(img_name)
                thread=ImportData(datasetid,update = True)
                thread.start()
                data['file name'] = file_name
                data['success']=True
                data['dataset name'] = datasetid
                data['status'] = 'QUEUED'
                                     
            except Exception as e:
                print(e)
                return jsonify(data)
        else:
                return jsonify(data)
    return jsonify(data)

if __name__ == 'main':
    print('Please wait until all models are loader')
    print('Load model')
    
    models_available = files.get_files_matching(settings.MODELS_ROOT)

    models = dict()
    
    #load the models to memory only once, when the app boots
    
    for path_to_model in models_available:
        file_name = os.path.basename(path_to_model)
        version_id = os.path.splitext(file_name)[0]
        models[version_id] = VggFace(path_to_model, is_origin = False)
    
    handler = ConcurrentRotatingFileHandler(settings.LOG_FILE, maxBytes=1024 * 1000 * 10, backupCount=5, use_gzip=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    app.logger.addHandler(handler)

    # https://stackoverflow.com/a/20423005/436721
    app.logger.setLevel(logging.INFO)

    
    app.run('0.0.0.0')
else:
    
    models_available = files.get_files_matching(settings.MODELS_ROOT)

    models = dict()
    
    #load the models to memory only once, when the app boots
    
    for path_to_model in models_available:
        file_name = os.path.basename(path_to_model)
        version_id = os.path.splitext(file_name)[0]
        models[version_id] = VggFace(path_to_model, is_origin = False)

    # disable logging so it doesn't interfere with testing
    app.logger.disabled = True

