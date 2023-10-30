from fastapi import FastAPI
import uvicorn
from fastapi import File, UploadFile
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import ultralytics
from ultralytics import YOLO
from distance_estimator import DistanceModelLoad
from distance_estimator import DistancePrediction
# image: np.ndarray
def predict(box):
    model = DistanceModelLoad()
    prediction = DistancePrediction(model, box)
    return prediction

def read(image):
    img = Image.open(BytesIO(image))
    return img
def preprocess(image: Image.Image):
    image = np.asfarray(image)
    return image

app = FastAPI()

@app.post('/detect')
async def image(file: bytes = File(...)):
    i = read(file)
    i = preprocess(i)
    detect_model = YOLO('weights/yolov8m.pt')
    results = detect_model(source=i, save=False)
    w = results[0].boxes.xywh.tolist()[0][2]
    h = results[0].boxes.xywh.tolist()[0][3]
    box = [w,h]
    predictions = predict(box)
    return 'distance: '+str(predictions)

uvicorn.run(app, port=8080, host='0.0.0.0')