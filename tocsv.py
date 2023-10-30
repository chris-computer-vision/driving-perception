import pandas as pd

def CarDatasetToCsv():
    dataset = pd.read_csv('data/annotations.csv')
    object_class = []
    width = []
    height = []
    distance = []
    for i in range(len(dataset)):
        if ( dataset['class'][i] == 'Car' ):
            object_class.append(dataset['class'][i])
            height.append( dataset['xmax'][i] - dataset['xmin'][i] )
            width.append( dataset['ymax'][i] - dataset['ymin'][i] )
            distance.append( dataset['zloc'][i] )

    dictionary = {'class': object_class, 'height': height, 'width': width, 'distance': distance}
    df = pd.DataFrame(dictionary)
    df.to_csv('data/car_train.csv')

def PeopleDatasetToCsv():
    dataset = pd.read_csv('data/annotations.csv')
    object_class = []
    width = []
    height = []
    distance = []
    for i in range(len(dataset)):
        if ( dataset['class'][i] == 'Pedestrian' or dataset['class'][i] == 'Person_sitting' ):
            object_class.append(dataset['class'][i])
            height.append( dataset['xmax'][i] - dataset['xmin'][i] )
            width.append( dataset['ymax'][i] - dataset['ymin'][i] )
            distance.append(dataset['zloc'][i])

    dictionary = {'class': object_class, 'height': height, 'width': width, 'distance': distance}
    df = pd.DataFrame(dictionary)
    df.to_csv('data/pedestrian_train.csv')