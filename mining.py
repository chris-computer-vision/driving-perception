import pandas as pd
import cv2
import json
f = open('data/prediction.json')
data = json.load(f)
if ( data['frame560'] == {} ):
    print('yes')
else: print('no')