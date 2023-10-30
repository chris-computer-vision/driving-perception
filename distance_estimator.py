import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data_path = 'data/car_train.csv'
train_csv = pd.read_csv(train_data_path)
train_mean = train_csv[['height', 'width', 'distance']].mean()
train_std = train_csv[['height', 'width', 'distance']].std()

class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, csvfile):
    self.datafile = pd.read_csv(csvfile)
    self.transform_data = self.datafile[['height', 'width', 'distance']]
    self.norm_data = (self.transform_data - train_mean[0:3]) / train_std[0:3]

  def __len__(self):
    return len(self.norm_data)

  def __getitem__(self, idx):
    data = self.norm_data.iloc[idx, 0:2].values
    label = self.norm_data.iloc[idx, 2:3].values

    data = data.astype('float32')
    label = label.astype('float32')

    data = torch.tensor(data)
    label = torch.tensor(label)

    return {'data': data, 'label': label}

def LoadTrainData():
    train_data = CustomDataset(train_data_path)
    dataset_length = len(train_data)
    train_data = torch.utils.data.DataLoader(dataset=train_data, shuffle=True, batch_size=512)
    return train_data, dataset_length

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.fc1 = nn.Linear(in_features=2, out_features=256, bias=True)
    self.fc2 = nn.Linear(in_features=256, out_features=128, bias=True)
    self.fc3 = nn.Linear(in_features=128, out_features=64, bias=True)
    self.fc4 = nn.Linear(in_features=64, out_features=32, bias=True)
    self.fc5 = nn.Linear(in_features=32, out_features=1, bias=True)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = self.fc5(x)
    return x

def DistanceModelTraining(train_data, dataset_length):
    net = Network()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    epoch_size = 100
    loss_data = []
    print('\nstart training:')
    for epoch in range(epoch_size):
        running_loss = 0.0
        
        for i, value in enumerate(train_data):
            inputs = value['data']
            labels = value['label']
            prediction = net(inputs)
            loss = criterion(prediction, labels)
            running_loss += loss.item() * inputs.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch: ', epoch, ' loss: ', running_loss/dataset_length)
        loss_data.append(running_loss/dataset_length)
    return loss_data, net 

def LossGraphPlotting(loss_data):
    plt.plot(loss_data)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss Graph')

def DistanceModelSave(net):
    torch.save(net.state_dict(), 'weights/car_distance.pth')

def DistanceModelLoad():
    model = Network()
    model.load_state_dict(torch.load('weights/car_distance.pth'))
    return model

def DistancePrediction(model, bounding_box):
    # bounding_box parameter is [height, width]
    bounding_box = (bounding_box - train_mean[0:2]) / train_std[0:2]
    bounding_box = bounding_box.astype('float32')
    bounding_box = torch.tensor(bounding_box)
    output = model(bounding_box)
    output = (output * train_std.values[-1] + train_mean.values[-1])
    return float(output)

def main():
    train_data, dataset_length = LoadTrainData()
    loss_data, net = DistanceModelTraining(train_data, dataset_length)
    LossGraphPlotting(loss_data)
    DistanceModelSave(net)

def test():
    model = DistanceModelLoad()
    out = DistancePrediction(model, [546,398])
    print(out)
if __name__ == "__main__":
   test()

# loss 0.1000 csv vehicle_train batch 1024 out_features 128 bias True layer 5 learningrate 0.0001 epoch 100
# loss 0.0875 csv vehicle_train batch 1024 out_features 128 bias True layer 5 learningrate 0.01 epoch 100
# loss 0.0863 csv vehicle_train batch 1024 out_features 128 bias True layer 5 learningrate 0.0001 epoch 1000
# loss 0.0956 csv vehicle_train batch 512 out_features 128 bias True layer 5 learningrate 0.0001 epoch 100
# loss 0.0923 csv vehicle_train batch 512 out_features 256 bias True layer 5 learningrate 0.0001 epoch 100
# loss 0.0888 csv vehicle_train batch 512 out_features 512 bias True layer 5 learningrate 0.0001 epoch 100
# loss 0.9175 csv vehicle_train batch 512 out_features 512 bias True layer 7 learningrate 0.0001 epoch 100