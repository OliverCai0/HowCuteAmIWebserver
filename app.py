from flask import Flask, redirect, url_for, request, jsonify
import json
import cv2
import math
import argparse
import numpy as np
import datetime
from HowCuteAmI.howcuteami import highlightFace, scale, cropImage
import base64
from flask_cors import CORS
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy import MetaData, Table, Column, Integer, String, Text, DateTime, inspect, insert, Float
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP
from sqlalchemy.orm import Session


load_dotenv()
import os

engine = create_engine(os.environ.get('DATABASE_URI'))
if not database_exists(engine.url):
    create_database(engine.url)

print(database_exists(engine.url))

if not inspect(engine).has_table("image"):
    print("Engine has detected that image table doesn't exist")
    print("Creating image table")
    metadata_obj = MetaData()
    image_table = Table('image', 
                    metadata_obj,
                    Column("image_id", Integer, primary_key=True),
                    Column("data", Text),
                    Column("score", Float),
                    Column("datetime", DateTime, default=func.now()))
    metadata_obj.create_all(engine)
print("Image table exists")
metadata_obj = MetaData(bind=engine)
image_table = Table('image', metadata_obj, autoload = True)

def readb64(encoded_data):
#    encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def evaluate(frame):
    scores = []
    faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")
    for faceBox in faceBoxes:
        # face detection net
        face = cropImage(frame, faceBox)
        face = cv2.resize(face, (224, 224))

        # beauty net
        blob=cv2.dnn.blobFromImage(face, 1.0/255, (224,224), MODEL_MEAN_VALUES, swapRB=False)
        beautyNet.setInput(blob)
        beautyPreds=beautyNet.forward()
        beauty=round(2.0 * sum(beautyPreds[0]), 1)
        # print(f'Beauty: {beauty}/10.0')
        scores.append(beauty)
    return scores

faceProto="HowCuteAmI/models/opencv_face_detector.pbtxt"
faceModel="HowCuteAmI/models/opencv_face_detector_uint8.pb"
ageProto="HowCuteAmI/models/age_googlenet.prototxt"
ageModel="HowCuteAmI/models/age_googlenet.caffemodel"
genderProto="HowCuteAmI/models/gender_googlenet.prototxt"
genderModel="HowCuteAmI/models/gender_googlenet.caffemodel"
beautyProto="HowCuteAmI/models/beauty_resnet.prototxt"
beautyModel="HowCuteAmI/models/beauty_resnet.caffemodel"

MODEL_MEAN_VALUES=(104, 117, 123)
#MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
color = (0,255,255)

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
beautyNet=cv2.dnn.readNet(beautyModel,beautyProto)

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    print("trying")

@app.route("/images", methods = ['GET','POST'])
def images():
    if request.method == 'POST':
        # print("Request", request.json.keys())
        t = request.json['data']
        frame = readb64(request.json['data'])
        scores = evaluate(frame)
        print(scores)
        if (len(scores) == 0):
            return jsonify({'scores': scores})
        # stmt = (insert(image_table).values(data=request.json['data']))
        query = image_table.insert().values(data=t, score= sum(scores) / len(scores))
        my_session = Session(engine)
        my_session.execute(query)
        my_session.commit()
        my_session.close()
        print("Inserted data point")
        return jsonify({'scores': scores})
    elif request.method == 'GET':
        print("Hit Get")
        session = Session(engine)
        query = session.query(image_table)
        results = query.all()
        results = [tuple(row) for row in results]
        # for row in results:
        #     print(row)
        return jsonify(results)
    
        
