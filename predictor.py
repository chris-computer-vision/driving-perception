from distance_estimator import DistanceModelLoad
from distance_estimator import DistancePrediction
from direction_estimator import DirectionPrediction
import ultralytics
from ultralytics import YOLO

groundtruth_size = (375, 1242)

def ObjectDetection(data, save, confidence):
    model = YOLO('weights/yolov8m.pt')

    results = model.track(source=data, save=save, conf=confidence)
    return results

def DistancePredictor(results):
    distance_model = DistanceModelLoad()
    distance = []

    for i in range(len(results)):

        for j in range(len(results[i].boxes.xyxy)):
            box = []
            box.append(results[i].boxes.xywh.tolist()[j][3])
            box.append(results[i].boxes.xywh.tolist()[j][2])

            if ( DistancePrediction(distance_model, box) < 0):
                distance.append(float(0))
            else:
                distance.append(DistancePrediction(distance_model, box))

    return distance

def DirectionPredictor(results):
    direction = []
    for i in range(len(results)):
        for j in range(len(results[i].boxes.xywh)):
            center_x = results[i].boxes.xywh.tolist()[j][0]
            center_y = results[i].boxes.xywh.tolist()[j][1]
            image_size = results[0].orig_shape
            direction.append(DirectionPrediction(center_x, center_y, image_size))
    return direction