import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from time import time
import os
from torch.utils.tensorboard import SummaryWriter
import utils

#Create folder for tensorboard's graph files
runs = 'runs'
os.makedirs(runs, exist_ok = True)

#set local directory path
dir_path = './'

#load data from pickled file
ck_datafile = open(dir_path + '/ck+_data.pickle', 'rb') #should this path not be recognised, source file path from submitted folder
ck_dataset = pickle.load(ck_datafile)
ck_datafile.close()

#Split and pre-process data
train_set,test_set = ck_dataset['training_data'], ck_dataset['test_data']
x_train, y_train = np.array(train_set[0]), np.array(train_set[1])
x_test, y_test = np.array(test_set[0]), np.array(test_set[1])

#Reshape images
x_train = x_train.reshape(4703, 100, 100)
x_test = x_test.reshape(1178, 100, 100)


#ToTensor
x_train = torch.from_numpy(x_train).unsqueeze(1).float()
y_train = torch.from_numpy(y_train).long().squeeze(1)
x_test = torch.from_numpy(x_test).unsqueeze(1).float()
y_test = torch.from_numpy(y_test_i).long().squeeze(1)

#Check images
# for i in range(5):
#   img = x_train_i.squeeze(1)[i, :, :]
#   plt.axis('off')
#   plt.imshow(img, cmap = 'Greys_r')
#   plt.show()

batch_size = 256
train_set = torch.utils.data.TensorDataset(x_train, y_train)
test_set = torch.utils.data.TensorDataset(x_test, y_test)
train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size = batch_size,
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, 
                                          batch_size = batch_size,
                                          shuffle = True)

#Define NN's architecture
class network_i(nn.Module):
    def __init__(self):
        super().__init__() 
        #initialize layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6,
                               kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12,
                               kernel_size = 3)
        self.dpout = nn.Dropout(0.2)        
        self.fc1 = nn.Linear(in_features = 12 * 94 * 94,
                              out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 8)
         
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dpout(x)     
        x = x.reshape(-1, 12 * 94 * 94)      
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Create model
model = network()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#Training loop
tb = SummaryWriter()
batch = next(iter(train_loader))
batch_test = next(iter(test_loader))
epochs = 30

for epoch in range(epochs):
  total_loss = 0
  total_correct_train = 0
  total_correct_test = 0
  time_zero = time()
  stacked = torch.tensor([])
  cm = torch.zeros(8, 8, dtype = torch.int32)

  for batch in train_loader_i:
    model_i.train()
    x, y = batch
    x = x.cuda()
    y = y.cuda()
    optimizer.zero_grad()
    y_hat = model_i(x)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    total_correct_train += get_num_correct(y_hat, y)
  
  for batch_test in test_loader_i:
    with torch.no_grad():
      model_i.eval()
      x_test_i, y_test_i = batch_test_i
      x_test_i = x_test_i.cuda()
      y_test_i = y_test_i.cuda()
      y_hat_test_i = model_i(x_test_i)
      total_correct_test += get_num_correct(y_hat_test_i, y_test_i)   
    stacked = torch.stack((y_test_i,
        y_hat_test_i.argmax(dim = 1)), 
        dim = 1)
    for i in stacked:
      j, k = i.tolist()
      cm[j, k] = cm[j, k] + 1
  
  accuracy_train = total_correct_train / len(train_set_i) 
  accuracy_test = total_correct_test / len(test_set_i)
  tb.add_scalars('Train & Test Accuracy', {'Train Accuracy': accuracy_train}, epoch)
  tb.add_scalars('Train & Test Accuracy', {'Test Accuracy': accuracy_test}, epoch)
  tb.add_scalar('Loss', total_loss, epoch)

  print('Epoch:', epoch, 'T&T time: %s min.'
        %(round((time() - time_zero)/60, 2)),
        'Loss: ', round(total_loss, 5), '\n',  'Train accuracy:',
        round(accuracy_train, 5), 'Test accuracy:', round(accuracy_test, 5),
        '\n')
tb.close()

#Plot Confusion Matrix
plt.figure(figsize = (10, 10))
plot_confusion_matrix(cm, names)

#Print F1-score per class
f1_score(names, cm, show = True)

#Get the nice graphs
#tensorboard --logdir=runs
