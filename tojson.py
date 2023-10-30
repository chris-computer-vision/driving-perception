import pandas as pd
import json

object_class = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
                19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
                26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
                32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
                37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
                43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
                50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
                57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
                63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
                69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
                76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

def ResultsToJson(results, distance, direction):
    frame_dict = {}
    count = 0

    for i in range(len(results)):
       box_dict = {}

       for j in range(len(results[i].boxes.xyxy)):
          box_dict['box'+str(j)] = {}

          if ( results[i].boxes.id is not None ):
              # add parameters here that you want to convert to json file
              box_dict['box' + str(j)]['xyxy'] = results[i].boxes.xyxy.tolist()[j]
              box_dict['box' + str(j)]['class'] = object_class[int(results[i].boxes.cls.tolist()[j])]
              box_dict['box' + str(j)]['confidence'] = results[i].boxes.conf.tolist()[j]
              box_dict['box' + str(j)]['trackid'] = int(results[i].boxes.id.tolist()[j])
              box_dict['box' + str(j)]['distance(m)'] = distance[count]
              box_dict['box' + str(j)]['direction(degrees)'] = direction[count]


          count += 1

       box_dict = {'frame'+str(i): box_dict}
       frame_dict.update(box_dict)

    with open("data/prediction.json", "w") as write_file:
        json.dump(frame_dict, write_file, indent=4)


