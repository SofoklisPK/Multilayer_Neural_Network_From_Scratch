import scipy.io as sio
import pickle
import numpy as np
import os
from time import time

import neural_network_class.py as nnClass

#set local directory path
DIR_PATH = './'
np.random.seed(42)

#Load the dataset from the files ~'train32_32.mat' and 'test32_32.mat'
#Update DIR_PATH when running locally so that it may find the necessary files
train_data = sio.loadmat(DIR_PATH + 'SVHN_dataset/train_32x32.mat')
test_data =  sio.loadmat(DIR_PATH + 'SVHN_dataset/test_32x32.mat')

#Flatten the 32x32 images into a single vector of 32x32x3 = 3072 features
x_train = train_data['X'].reshape(32*32*3,73257).T
#Normalize/Standardize the input data
train_mean = np.atleast_2d(np.mean(x_train, axis = 1)).T
x_train = (x_train- train_mean)/train_mean
#Replace the label '10' with '0'
y_train = train_data['y']%10

#Flatten the 32x32 images into a single vector of 32x32x3 = 3072 features
x_test = test_data['X'].reshape(32*32*3,26032).T
#Normalize/Standardize the input data
test_mean = np.atleast_2d(np.mean(x_test, axis = 1)).T
x_test = (x_test- test_mean)/test_mean
#Replace the label '10' with '0'
y_test = test_data['y']%10

size_of_train_sample = len(y_train)
size_of_test_sample = len(y_test)


#Set NN hyperparameters
totalEpochs = 10
myLayerSizes = [3072,128,128,10]
myLayerActivations = ['relu','relu','softmax']
myLearningRate = 0.0001
myDropoutRate = 0.2
myBias = True
myLamda = 0.01
myBatchSize = 64
myBeta = 0.9


### ONLY RUN WHEN INITIALIZING THE NEURAL NETWORK ###
# !careful not to overwrite older versions!

modelName = 'myNN__CK_onSVHN'

### change this manually if you wish to overwrite existing saved model
overwrite_model = False
### ^^^^

#check to see if model already exists or if model exists and we want to overwrite
if (os.path.exists(DIR_PATH+modelName) == False) or overwrite_model:
    if overwrite_model: print ('This model exists... Overwriting model now!')
    #Create a new NN model based on SVHN_NN class (Three layers, Relu hidden and softmax output)
    myNN = nnClass.CK_NN(layer_sizes = myLayerSizes, layer_activations = myLayerActivations, learning_rate = myLearningRate, dropout_prob=myDropoutRate, bias = myBias, lamda = myLamda, beta = myBeta)
    print('{0} Hidden Nodes, LR = {1}, dropout = 0.2, {2} epochs, {3} bias, {4} batch size, lamda = {5}, beta = {6}'.format(myLayerSizes[1:-1],myLearningRate,totalEpochs,myBias,myBatchSize,myLamda,myBeta))
    #Save initialized model to file
    myPickle = open(DIR_PATH+modelName, 'wb')
    nnData = {'NN': myNN,
            'epoch': 0,
            'test_accuracies': [],
            'train_accuracies': [],
            'training_loss' : []}
    pickle.dump(nnData, myPickle)
    myPickle.close()
#if model exists and we don't want to overwrite
else:
    print('This model already exists!!')



### HERE, WE TRAIN THE NEURAL NETWORK! ###
time0 = time()

#Fuction to test the accuracy of the myNN with test set 'x_test' & labels 'y_test'
def test_accuracy(cnt = 0):
    print(' testing for accuracy...')

    #predict labels for whole dataset 'x_test' (passes all testing samples together into the NN)
    pred_y = trainedNN.run(x_test.T)
    pred_y = np.atleast_2d(np.argmax(pred_y, axis = 1))
    #Sum up all instances where the predicted label was the same as the target label
    num_of_correct_test = np.sum((pred_y.T == y_test))

    #predict labels for whole dataset 'x_train' (passes all training samples together into the NN)
    pred_y = trainedNN.run(x_train.T)
    pred_y = np.atleast_2d(np.argmax(pred_y, axis = 1))
    #Sum up all instances where the predicted label was the same as the target label
    num_of_correct_train = np.sum((pred_y.T == y_train))

    #print(' @',cnt,' samples : number of correct predictions: ', num_of_correct, ' out of ', size)
    print(' @',cnt,' epochs : Training accuracy : {0:.2f}%'.format(num_of_correct_train*100/size_of_train_sample))
    print(' @',cnt,' epochs : Testing accuracy : {0:.2f}%'.format(num_of_correct_test*100/size_of_test_sample))
    return num_of_correct_train/size_of_train_sample, num_of_correct_test/size_of_test_sample

#Load the neural network from pickle file
modelName = 'myNN_CK_onSVHN'
myPickle = open(DIR_PATH+modelName, 'rb')
nnData = pickle.load(myPickle)
myPickle.close()
trainedNN = nnData['NN']          #wont be trained on first call
trainedEpochs = nnData['epoch']   #equal to 0 on first call
trainAccuracies = nnData['train_accuracies'] #is [] on first call
testAccuracies = nnData['test_accuracies'] #is [] on first call
trainingLoss = nnData['training_loss'] #is [] on first call

print('{0} Hidden Nodes, LR = {1:f}, dropout = 0.2, {2} epochs, {3} bias, {4} batch size, lamda = {5}, beta = {6}'.format(myLayerSizes[1:-1],myLearningRate,totalEpochs,myBias,myBatchSize,myLamda,myBeta))

#train over many epochs
for e in range(trainedEpochs,totalEpochs):
    print('\nTraining Epoch ', e+1, end='\t ')
    tmp = 0
    running_losses = []

    #Iterate batch-wise through all the samples of the training dataset
    for j in range(0,size_of_train_sample,myBatchSize):

        #set start and end indexes of the batch
        start_idx = j
        end_idx = min(j+myBatchSize,size_of_train_sample)

        #Prettify the process of training (prints '-' every 1000 batches)
        if(j//1000==tmp):
            print('-',end = '')
            tmp += 1

        #Load the samples and change the format of yi from scalar (the labels) into a onehot array of size (batchSize,10)
        xi, yi = x_train[start_idx:end_idx], y_train[start_idx:end_idx]
        yi_array = np.zeros((end_idx-start_idx,10))
        yi_array[np.arange(end_idx-start_idx),yi.T] = 1

        #train the model
        batch_loss = trainedNN.train(xi, yi_array, size_of_train_sample)
        running_losses.append(batch_loss)

    avg_running_loss = np.average(running_losses)
    trainingLoss.append(avg_running_loss)

    print("\nTraining Time (in minutes) = {0:.2f} min".format((time()-time0)/60))
    train_accur, test_accur = test_accuracy(e+1)
    trainAccuracies.append(train_accur)
    testAccuracies.append(test_accur)
    print('@', e+1, 'epochs : Training Loss : ', avg_running_loss)

    #Store the training of the model on pickle file
    nnData = {'NN': trainedNN,
              'epoch': e,
              'train_accuracies': trainAccuracies,
              'test_accuracies' : testAccuracies,
              'training_loss' : trainingLoss}
    myPickle = open(DIR_PATH+modelName, 'wb')
    pickle.dump(nnData, myPickle)
    myPickle.close()
