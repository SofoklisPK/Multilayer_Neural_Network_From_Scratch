#all import packages
from scipy.stats import truncnorm
import numpy as np

np.random.seed(42)

def truncated_normal(mean = 0, sd = 1, low = 0, upp = 10):
    return truncnorm((low - mean)/sd,
                     (upp - mean)/sd,
                     loc = mean,
                     scale = sd)


##### Layer Class #####

class Activated_Linear_Layer:

    #Initialization of the parameters of the Layer
    def __init__(self, num_of_inputs, num_of_outputs, activation = 'relu', bias = True, lamda = 0, beta = 0.9):
        self.bias = bias
        self.lamda = lamda
        self.activation = activation    # Either 'relu' or 'softmax'
        rad = 1/(np.sqrt(num_of_inputs + self.bias))
        truncate = truncated_normal(mean = 0, sd = 1, low = -rad, upp = rad)
        #random initialization of weights based off of truncnorm (use seed 42 for testing purposes)
        # and if we have biases, add one to the dimension to simulate bias node
        self.weights = truncate.rvs((num_of_outputs, num_of_inputs + self.bias))                        #(W1) & (W2)
        self.input_array_bnodes = []
        self.beta = beta
        self.velocity = np.zeros(self.weights.shape)

    #Forward propogation through the ReLU Layer
    def forward_pass(self, input_array):
        #if we are using biases then add a "pseudo-node" with values 1, to the input to work as the bias
        if self.bias:
            bias_nodes = np.tile(1,(input_array.shape[0],1))
            self.input_array_bnodes= np.concatenate((input_array, bias_nodes), axis = 1)                #(X+) & (A1+)
        else:
            self.input_array_bnodes = input_array                                                       # or just (X) & (A1)
        self.output_array= np.dot(self.input_array_bnodes, self.weights.T)                              #(Z1 = X.W1t) & (Z2 = A1.W2t)
        #ReLU activation of the linear output
        if self.activation == 'relu':
            activated_output_array= np.maximum(self.output_array, 0.0)                                  #A1 = ReLU(Z1)
        #Softmax activation of the linear output
        elif self.activation == 'softmax':
            exp_array = np.exp(self.output_array)
            activated_output_array = exp_array / np.sum(exp_array, axis = 1)[:,np.newaxis]              #A2 = Softmax(Z2)  corrected so that it does proper softmax when workling with mini batches
        return activated_output_array

    #Backward propogation through the ReLU Layer and weight updates
    def backward_pass (self, delta_activated_output, learning_rate = 0.01):                                                   #(dA2) & (dA1)
        m = delta_activated_output.shape[0]                                                             #m = batch size

        #Derivative of ReLU activation fuction
        if self.activation == 'relu':
            delta_output = np.multiply(delta_activated_output, (self.output_array>= 0))                 #(dZ1 = ReLU'(dA1))
        #Derivative of the Softmax acivation function (This is not calculated from the cost function, but straight from dZ = y_hat - y)
        elif self.activation == 'softmax':
            delta_output = delta_activated_output                                                       #(dZ2 = dA2 = y_hat - y)
        #Derivative of input to pass on to previous layers
        delta_input_array = np.dot(delta_output, self.weights)                                          #(dA1 = dZ2.W2) & (dX = dZ1.W1)
        if self.bias: delta_input_array = delta_input_array[:,:-1]                                      #if we have a bias node, the we have extra on dA
        #Derivative of weights, in order to update, with L2 regularization and momentum
        delta_weights = np.dot(delta_output.T,self.input_array_bnodes) + (self.lamda/m)*self.weights    #(dW1 = dZ1.Xt + (lamda/m)*W1) & (dW2 = dZ2.A1t + (lamda/m)*W2)
        self.velocity = self.beta*self.velocity + (1-self.beta)*delta_weights                           #(V[t] = beta*V[t-1] + (1-beta)*dW)
        self.weights -= learning_rate*self.velocity                                                     #(W1 = W1 - a*V) & (W2 = W2 - a*dV) #check to see if V is common over each layer, or seperate.. here it is applied diff per layer

        return delta_input_array



##### Neural Network Class #####

class CK_NN :

    #Initialization of the Neural Network with 3 layer (Input, Hidden-ReLU, Output-Softmax)
    def __init__ (self, layer_sizes = [3072,128,10], layer_activations = ['relu', 'softmax'], dropout_prob = 0.2, bias = True, lamda = 0, beta = 0.9):

        #we assume that len(layer_sizes) = len(layer_activations) + 1... We should normally check, so as to make sure that the NN is created correctly..
        self.layers = []
        for i in range(len(layer_activations)):
            self.layers.append(Activated_Linear_Layer(num_of_inputs =layer_sizes[i],
                                              num_of_outputs = layer_sizes[i+1],
                                              activation = layer_activations[i],
                                              bias = bias,
                                              lamda = lamda,
                                              beta = beta))
        self.dropout_prob = dropout_prob
        self.lamda = lamda

    #Forward run through the whole NN (predict y_hat)
    def run (self, input_array, dropout = False):
        input_array = np.array(input_array).T

        layer_activated_output = input_array
        for i in range(len(self.layers)):
            layer_activated_output = self.layers[i].forward_pass(layer_activated_output)
            #dont apply dropout if in testing accuracy, or when in final layer (i.e. softmax layer)
            if dropout and (i < len(self.layers) - 1):
                mask = (np.random.random_sample(layer_activated_output.shape) > self.dropout_prob) /(1 - self.dropout_prob)
                layer_activated_output *= mask
        return layer_activated_output

    #Train the NN on a specific input and target labels
    def train (self, input_array, target_array, size_of_train_sample = 100, learning_rate = 0.01):
        in_array = np.atleast_2d(np.array(input_array))
        tar_array = np.atleast_2d(np.array(target_array))

        #calculate the predicted output of the NN
        pred_output = self.run(in_array.T, dropout = True)
        #Calculate the error of the predicted output
        #l2_cost = (self.lamda/(2*input_array.shape[0]))*(np.sum(np.square(self.hidden_layer.weights))+np.sum(np.square(self.output_layer.weights)))
        #we don't apply L2 here because we don't manually calculate cost and delta_cost, but skip straight to dZ = y - y_hat
        delta_output =  pred_output - tar_array #+ l2_cost
        mse_loss = np.average(np.square(delta_output))

        #Back propogate through the NN and update weights
        delta_activated_output = delta_output
        for i in range(len(self.layers)-1,-1,-1):
            delta_activated_output = self.layers[i].backward_pass(delta_activated_output, learning_rate)

        return mse_loss

    #Print the values of the weights for each layer
    def print_weights(self):
        for i in range(len(self.layers)):
            print('layer ', i, 'weights: \n', self.layers[i].weights)
