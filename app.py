from flask import Flask, redirect, url_for, request, jsonify
import json
import cv2
import math
import argparse
import numpy as np
from HowCuteAmI.howcuteami import highlightFace, scale, cropImage
import base64
from flask_cors import CORS

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

@app.route("/images", methods = ['POST'])
def images():
    if request.method == 'POST':
        # print("Request", request.json.keys())
        frame = readb64(request.json['data'])
        scores = evaluate(frame)
        print(scores)
        return jsonify({'scores': scores})
        
