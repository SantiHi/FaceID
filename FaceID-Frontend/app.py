from flask import Flask, render_template, Response
import face_recognition
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import metrics
import pickle
from keras_facenet import FaceNet
app=Flask(__name__)

detected_persons = [] # list of names who were EVER detected in the frame. 

def filePaths(directory):
    files = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            files.append(os.path.join(dirpath, f))
    return files

def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot

with open('embeds.npy', 'rb') as f:
    X = np.load(f)

with open('train_labels.npy', 'rb') as f:
    y = np.load(f)

labels = ['Abhisheik Sharma', 'Akash Wudali', 'Akshat Alok', 'Ashwin Pulla', 'Ayaan Siddiqui', 'Daniel Qiu', 'Darren Kao', 'Dev Kodre', 'Emi Zhang', 'Grace Liu', 'Jesse Choe', 'Krish Malik', 'Lucas Marschoun', 'Manav Gagvani', 'Matthew Palamarchuk', 'Mihika Dusad', 'Om Gole', 'Pranav Kuppa', 'Pranav Panicker', 'Pranav Vadde', 'Preston Brown', 'Raghav Sriram', 'Rohan Kalahasty', 'Samarth Bhargav', 'Santiago Criado', 'Shreyan Dey', 'Sritan Motati', 'Tanvi Pedireddy', 'Tejesh Dandu', 'Vishal Nandakumar']


#YOLO CODE
def capture_by_frames(): 
    model = YOLO('best.pt')
    embedder = FaceNet()
    global camera
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()

        if not success: continue

        results = model(frame)
        coords = results[0].boxes.xyxy.numpy()

        if results[0].boxes.xyxy.nelement() == 0: 
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        if coords.any() == None: continue
        
        b = np.array([coords[i].astype(int) for i in range(coords.shape[0])])

        face_images = np.array([cv2.resize(frame[i[1]:i[3], i[0]:i[2]], (64, 64)) for i in b])
        embedding_results = embedder.embeddings(face_images)


        y_pred = []
        for res in embedding_results:
            face_distances = face_recognition.face_distance(X, res)
            y_pred.append(y[np.argmin(face_distances)])

        names = [labels[i] for i in y_pred]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0) 
        thickness = 1
        for i in range(len(names)):
            org = (b[i, 0], b[i, 1])
            name = names[i]
            for name in names:
                if name not in detected_persons:
                    detected_persons.append(name)
            annotated_frame = cv2.putText(frame, name, org, font, fontScale, color, thickness, cv2.LINE_AA) 

        # cv2.imshow("YOLOv8 Inference", annotated_frame)

        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        # Display the resulting image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start',methods=['POST'])
def start():
    return render_template('index.html')

@app.route('/stop',methods=['POST'])
def stop():
    if camera.isOpened():
        camera.release()
    return render_template('stop.html', detected_persons=detected_persons)

@app.route('/video_capture')
def video_capture():
    return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True,use_reloader=False, port=8000)