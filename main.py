from predictor import ObjectDetection
from distance_estimator import DistanceModelLoad
from distance_estimator import DistancePrediction 
from direction_estimator import DirectionPrediction
from tojson import ResultsToJson
from predictor import DistancePredictor
from predictor import DirectionPredictor

def main():
    video = 'input/hkvideo3.mp4'
    save = True
    confidence = 0.7
    results = ObjectDetection(video, save, confidence)
    distance = DistancePredictor(results)
    direction = DirectionPredictor(results)
    ResultsToJson(results, distance, direction)

if __name__ == "__main__":
    main()