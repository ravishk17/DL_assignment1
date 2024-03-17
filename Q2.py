#!/usr/bin/env python
# coding: utf-8

# In[6]:


from keras.datasets import fashion_mnist
import math
import numpy as np

def softMax_q2(aL):
    den=0
    aL = np.clip(aL,-499,499)
    for i in range(10):
        den+=np.exp(aL[0][i])
    y = np.zeros(10)
    for i in range(10):
        y[i] = np.exp(aL[0][i])/den
    return y
    
def forward_pass(X,hiddenLayers,bias,weightMatrix):
    probability=[]
    for x in X:
        input = x.reshape(1,-1).T
        for i in range(1,hiddenLayers+1):
            a_i = bias[i].T + weightMatrix[i]@input
            h_i = np.tanh(a_i)
            input=h_i
        a_l = bias[hiddenLayers+1].T+weightMatrix[hiddenLayers+1]@h_i
        out = softMax_q2(a_l.T)
        probability.append(out)
    return probability

# 2nd question
def q2():
    neuronPerHiddenLayer = [16,16,16]
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    inputLayerNeuron = train_images.shape[1]*train_images.shape[2]
    outputLayerNeuron = 10
    train_images = train_images/255.0
    neuronPerLayer = [inputLayerNeuron] + neuronPerHiddenLayer + [outputLayerNeuron]
    weightMatrix = [0]
    bias = [0]
    for j in range(1,len(neuronPerLayer)):
        limit = np.sqrt(2/float(neuronPerLayer[j] + neuronPerLayer[j-1]))
        weightMatrix.append(np.random.normal(0.0,limit,size = (neuronPerLayer[j],neuronPerLayer[j-1])))
        bias.append(np.zeros((1,neuronPerLayer[j])))
    hiddenLayers = len(neuronPerHiddenLayer)
    
    answers = forward_pass(train_images,hiddenLayers,bias,weightMatrix)
    np.set_printoptions(suppress=True)
    print(answers[0])
    print(sum(answers[0]))
q2()


# In[ ]:




