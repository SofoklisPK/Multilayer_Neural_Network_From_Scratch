import pickle
import numpy as np
np.random.seed(42)

ck_datafile = open('./ck_data.pickle', 'rb')
ck_dataset = pickle.load(ck_datafile)
ck_datafile.close()

train_data,test_data, img_shape = ck_dataset['training_data'], ck_dataset['test_data'], ck_dataset['img_dim']


train_feat, train_labels = train_data[0], np.array(train_data[1])
test_feat, test_labels = test_data[0], np.array(test_data[1])

np.random.shuffle(train_feat)
np.random.shuffle(train_labels)
np.random.shuffle(test_feat)
np.random.shuffle(test_labels)

#print distribution of labels for the training set and testing set
print('Training dataset')
for i in range(8):
    print('{0}: '.format(i),sum(train_labels == i), '({0}%)'.format(sum(train_labels == i)*100/len(train_labels)))
print('\nTesting dataset')
for i in range(8):
    print('{0}: '.format(i),sum(test_labels == i), '({0}%)'.format(sum(test_labels == i)*100/len(test_labels)))


size_of_train_sample = train_feat.shape[0]
size_of_test_sample = test_feat.shape[0]


#Set NN hyperparameters
totalEpochs = 500
myLayerSizes = [10000,512,512,8]
myLayerActivations = ['relu','relu','softmax']
myLearningRate = 0.0005
myDropoutRate = 0.2
myBias = True
myLamda = 10
myBatchSize = 128
myBeta = 0.9

#set local directory path
DIR_PATH = './'



### ONLY RUN WHEN INITIALIZING THE NEURAL NETWORK ###
# !careful not to overwrite older versions!
import os

modelName = 'myNN_CK'

### change this manually if you wish to overwrite existing saved model
overwrite_model = False
### ^^^^

#check to see if model already exists or if model exists and we want to overwrite
if (os.path.exists(DIR_PATH+modelName) == False) or overwrite_model:
    if overwrite_model: print ('This model exists... Overwriting model now!')
    #Create a new NN model based on SVHN_NN class (Three layers, Relu hidden and softmax output)
    myNN = CK_NN(layer_sizes = myLayerSizes, layer_activations = myLayerActivations, dropout_prob=myDropoutRate, bias = myBias, lamda = myLamda, beta = myBeta)
    print('{0} Hidden Nodes, LR = {1}, dropout = 0.2, {2} epochs, {3} bias, {4} batch size, lamda = {5}, beta = {6}'.format(myLayerSizes[1:-1],myLearningRate,totalEpochs,myBias,myBatchSize,myLamda,myBeta))
    #Save initialized model to file
    myPickle = open(DIR_PATH+modelName, 'wb')
    nnData = {'NN': myNN,
            'epoch': 0,
            'test_accuracies': [],
            'train_accuracies': [],
            'training_loss': [],
            'f1_measure' : []}
    pickle.dump(nnData, myPickle)
    myPickle.close()
#if model exists and we don't want to overwrite
else:
    print('This model already exists!!')



    ### HERE, WE TRAIN THE NEURAL NETWORK! ###

from time import time
from sklearn.metrics import f1_score
import pickle
np.random.seed(42)
time0 = time()

#Function to test the accuracy of the myNN with test set 'x_test' & labels 'y_test'
def test_accuracy(cnt = 0):
    print(' testing for accuracy...')

    #predict labels for whole dataset 'x_test' (passes all testing samples together into the NN)
    pred_y = trainedNN.run(test_feat.T)
    pred_y = np.atleast_2d(np.argmax(pred_y, axis = 1)).T
    #Sum up all instances where the predicted label was the same as the target label
    num_of_correct_test = np.sum((pred_y == test_labels))


    f1 = f1_score(test_labels,pred_y,average='weighted')
    #print out the distribution of predicted labels for the test set
    for i in range(8):
        print('{0}: '.format(i),sum(pred_y == i), '({0}%)'.format(sum(pred_y == i)*100/len(pred_y)))

    #predict labels for whole dataset 'x_train' (passes all training samples together into the NN)
    pred_y = trainedNN.run(train_feat.T)
    pred_y = np.atleast_2d(np.argmax(pred_y, axis = 1)).T
    #Sum up all instances where the predicted label was the same as the target label
    num_of_correct_train = np.sum((pred_y == train_labels))

    #print(' @',cnt,' samples : number of correct predictions: ', num_of_correct, ' out of ', size)
    print(' @',cnt,' epochs : Training accuracy : {0:.2f}%'.format(num_of_correct_train*100/size_of_train_sample))
    print(' @',cnt,' epochs : Testing accuracy : {0:.2f}%'.format(num_of_correct_test*100/size_of_test_sample))
    print(' @',cnt,' epochs : F1 measure : {0:.2f}'.format(f1))

    return num_of_correct_train/size_of_train_sample, num_of_correct_test/size_of_test_sample, f1



#Load the neural network from pickle file
modelName = 'myNN_CK'
myPickle = open(DIR_PATH+modelName, 'rb')
nnData = pickle.load(myPickle)
myPickle.close()
trainedNN = nnData['NN']          #wont be trained on first call
trainedEpochs = nnData['epoch']   #equal to 0 on first call
trainAccuracies = nnData['train_accuracies'] #is [] on first call
testAccuracies = nnData['test_accuracies'] #is [] on first call
trainingLoss = nnData['training_loss'] #is [] on first call
f1Measure = nnData['f1_measure'] #is [] on first call

print('{0} Hidden Nodes, LR = {1}, dropout = {2}, {3} epochs, {4} bias, {5} batch size, lamda = {6}, beta = {7}'.format(myLayerSizes[1:-1],myLearningRate,myDropoutRate,totalEpochs,myBias,myBatchSize,myLamda,myBeta))

curr_learning_rate = myLearningRate

#train over many epochs
for e in range(trainedEpochs,totalEpochs+1):
    print('\nTraining Epoch ', e+1, end='\t ')
    tmp = 0
    running_mse_losses = []

    #learning rate update schedule where the learning rate is divided by two every ten epochs
    if (e%10 == 0):
        curr_learning_rate = curr_learning_rate/2

    #Iterate batch-wise through all the samples of the training dataset
    for j in range(0,size_of_train_sample,myBatchSize):

        #set start and end indexes of the batch
        start_idx = j
        end_idx = j+myBatchSize
        end_idx = min(end_idx,size_of_train_sample)

        #Prettify the process of training (prints '-' every 1000 batches)
        if(j//1000==tmp):
            print('-',end = '')
            tmp += 1

        #Load the samples and change the format of yi from scalar (the labels) into a onehot array of size (batchSize,10)
        xi, yi = train_feat[start_idx:end_idx], train_labels[start_idx:end_idx]
        yi_array = np.zeros((end_idx-start_idx,8))
        yi_array[np.arange(end_idx-start_idx),yi.T] = 1

        #train the model
        batch_mse_loss = trainedNN.train(xi, yi_array, size_of_train_sample, curr_learning_rate)
        running_mse_losses.append(batch_mse_loss)

    avg_running_loss = np.average(running_mse_losses)
    trainingLoss.append(avg_running_loss)


    print("\nTraining Time (in minutes) = {0:.2f} min".format((time()-time0)/60))
    #every 5 epochs, print out accuracies
    if (e%5 == 0):
        train_accur, test_accur, f1 = test_accuracy(e+1)
        trainAccuracies.append(train_accur)
        testAccuracies.append(test_accur)
        f1Measure.append(f1)
    print('@', e+1, 'epochs : training loss : {0:.6f}'.format(avg_running_loss))

    #Store the training of the model on pickle file
    nnData = {'NN': trainedNN,
              'epoch': e,
              'train_accuracies': trainAccuracies,
              'test_accuracies' : testAccuracies,
              'training_loss' : trainingLoss,
              'f1_measure' : f1Measure}
    myPickle = open(DIR_PATH+modelName, 'wb')
    pickle.dump(nnData, myPickle)
    myPickle.close()
