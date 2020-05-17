import socketio
import eventlet
from flask import Flask
import numpy as np
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()

app = Flask(__name__) #'__main__' name 
speed_limit = 10
def img_preprocess(img): 
    img = img[60:135,:,:] #shortening height of image [height, width, layer]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) #image space used for training NVIDIA neural model 
    img = cv2.GaussianBlur(img, (3,3), 0) #smoothening image technique
    img = cv2.resize(img, (200,66)) #resizing image as per specifications of NVIDIA neural model 
    img = img/255 #normalizing image (reduce variance btw image data without visual impact)
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed']) #accessing speed
    image = Image.open(BytesIO(base64.b64decode(data['image']))) #image is base64 encoded
    image = np.asarray(image) #transform the image data into a numpy array
    image = img_preprocess(image) #apply defined preprocess function to the image
    image = np.array([image]) #add extra layer to image array?
    steering_angle = float(model.predict(image)) #model predicts a steering angle
    throttle = 1.0 - speed/speed_limit #adjust throttle based on speed data
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print('I am connected to the Udacity Simulator!')
    send_control(0,0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data ={
        'steering_angle': steering_angle.__str__(), 
        'throttle': throttle.__str__()

    })

if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

############  
