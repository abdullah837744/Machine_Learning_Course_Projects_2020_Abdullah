# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:28:02 2023

@author: abdullah
"""

#!/usr/bin/env python
# coding: utf-8

# # Batch Gradient Descent

# In[39]:


print('#######################################################################')
print("Batch_gradient_descent")
print("       ")
print('#######################################################################')


# In[40]:


import numpy as np
import sys
import pandas as pd


# In[41]:


f = open(sys.argv[1])
traindata = np.loadtxt(f)
train = traindata[:,1:]
trainlabels = traindata[:,0]
onearray = np.ones((train.shape[0],1))
train = np.append(train,onearray,axis=1)
rows = train.shape[0]
cols = train.shape[1]


# In[42]:


f = open(sys.argv[2])
testdata = np.loadtxt(f)
test = testdata[:,1:]
testlabels = testdata[:,0]
onearray = np.ones((test.shape[0],1))
test = np.append(test,onearray,axis=1)


# In[43]:


##sigmoid function
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


# In[44]:


def back_prop(traindata, trainlabels):
    hidden_nodes = 3
    W = np.random.rand(hidden_nodes, cols)
    w = np.random.rand(hidden_nodes)
    hidden_layer = np.matmul(train, np.transpose(W))
    hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
    output_layer = np.matmul(hidden_layer, np.transpose(w))
    obj = np.sum(np.square(output_layer - trainlabels))
    stop=0.001
    epochs = 1000
    eta = .001
    prevobj = np.inf
    i=0
    while(prevobj - obj > stop and i < epochs ):
        prevobj = obj
        dellw = (np.dot(hidden_layer[0,:],w)-trainlabels[0])*hidden_layer[0,:]
        for j in range(1, rows):
            dellw += (np.dot(hidden_layer[j,:],np.transpose(w))-trainlabels[j])*hidden_layer[j,:]
        w = w - eta*dellw
        dells = (np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[0] * (hidden_layer[0,0])*(1-hidden_layer[0,0])*train[0]
        dellu = (np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[1] * (hidden_layer[0,1])*(1-hidden_layer[0,1])*train[0]
        dellv = (np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[2] * (hidden_layer[0,2])*(1-hidden_layer[0,2])*train[0]
        for j in range(1, rows):
            dells += (np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[0] * (hidden_layer[j,0])*(1-hidden_layer[j,0])*train[j]
            dellu += (np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[1] * (hidden_layer[j,1])*(1-hidden_layer[j,1])*train[j]
            dellv += (np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[2] * (hidden_layer[j,2])*(1-hidden_layer[j,2])*train[j]
        dellW = np.array([dells, dellu, dellv])
        for k in range(3):
            W[k] = W[k] - eta*dellW[k]
        hidden_layer = np.matmul(train, np.transpose(W))
        hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
        output_layer = (np.matmul(hidden_layer, np.transpose(w)))
        obj = np.sum(np.square(output_layer - trainlabels))
        i = i + 1
        print("i=",i)
        print("Objective=",obj)
    print("Final_W=",W)
    print("Final_w=",w)
    return [W, w]


# In[45]:


[W,w]=back_prop(train, trainlabels)
predict_hidden = sigmoid(np.matmul(test, np.transpose(W)))
back_prop_predictions =np.sign(np.matmul(predict_hidden,np.transpose(w)))
print("   ")
print("Predictions_test_label_back_prop=",back_prop_predictions)
correct = 0
for i in range(len(testlabels)):
    if testlabels[i] ==back_prop_predictions[i]:
        correct += 1
    accuracy=correct / float(len(testlabels)) * 100.0
print("   ")
print("Accuracy of Batch gradient=",accuracy)


# In[46]:


OUT= open("back_prop_predictions", "w") 
content = str(back_prop_predictions) 
OUT.write(content) 
OUT.close() 


# # Stocastic Gradient Descent

# In[9]:


print('#######################################################################')
print("Stocastic_gradient_descent")
print("       ")
print('#######################################################################')


# In[10]:


def sgd(traindata,trainlabels,batch_size,epochs):
    hidden_nodes = 3
    w = np.random.rand(hidden_nodes)-0.5
    W = np.random.rand(hidden_nodes, cols)-0.7
    eta = 0.001
    for i in range(0,epochs):
        print("*********************************************************")
        print("epochs=",i)
        print("*********************************************************")
        np.random.shuffle(traindata)
        no_batches=len(train)//batch_size
        for j in range(1,no_batches):
            print("   ")
            print("no.of batches=",j)
            print("   ")
            for k in range(j*batch_size,(j+1)*batch_size):
                train_s = traindata[:k,1:]
                onearray = np.ones((train_s.shape[0],1))
                train_s = np.append(train_s,onearray,axis=1)
                trainlabel_s = traindata[:k,0]
                hidden_layer_mb = np.matmul(train_s, np.transpose(W))
                hidden_layer_mb = np.array([sigmoid(xi) for xi in hidden_layer_mb])
                output_layer = np.matmul(hidden_layer_mb, np.transpose(w))
                obj = np.sum(np.square(output_layer - trainlabel_s))
                dellw = (np.dot(hidden_layer_mb[0,:],w)-trainlabel_s[0])*hidden_layer_mb[0,:]
                for j in range(1, batch_size):
                    dellw += (np.dot(hidden_layer_mb[j,:],np.transpose(w))-trainlabels[j])*hidden_layer_mb[j,:]
                w = w - eta*dellw
                dells = np.sum(np.dot(hidden_layer_mb[0,:],w)-trainlabels[0])*w[0] * (hidden_layer_mb[0,0])*(1-hidden_layer_mb[0,0])*train[0]
                dellu = np.sum(np.dot(hidden_layer_mb[0,:],w)-trainlabels[0])*w[1] * (hidden_layer_mb[0,1])*(1-hidden_layer_mb[0,1])*train[0]
                dellv = np.sum(np.dot(hidden_layer_mb[0,:],w)-trainlabels[0])*w[2] * (hidden_layer_mb[0,2])*(1-hidden_layer_mb[0,2])*train[0]
                for j in range(1,batch_size):
                    dells += np.sum(np.dot(hidden_layer_mb[j,:],w)-trainlabels[j])*w[0] * (hidden_layer_mb[j,0])*(1-hidden_layer_mb[j,0])*train[j]
                    dellu += np.sum(np.dot(hidden_layer_mb[j,:],w)-trainlabels[j])*w[1] * (hidden_layer_mb[j,1])*(1-hidden_layer_mb[j,1])*train[j]
                    dellv += np.sum(np.dot(hidden_layer_mb[j,:],w)-trainlabels[j])*w[2] * (hidden_layer_mb[j,2])*(1-hidden_layer_mb[j,2])*train[j]
                    dellW = np.array([dells, dellu, dellv])
                for k in range(3):
                    W[k] = W[k] - eta*dellW[k]
                hidden_layer_mb = np.matmul(train, np.transpose(W))
                hidden_layer_mb = np.array([sigmoid(xi) for xi in hidden_layer_mb])
                output_layer = (np.matmul(hidden_layer_mb, np.transpose(w)))
                obj = np.sum(np.square(output_layer - trainlabels))
                print("Objective=",obj)
    print("Final_W=",W)
    print("Final_w=",w)
    return [W,w]


# In[11]:


[W1,w1]=sgd(traindata, trainlabels,5,100)
predict_hidden = sigmoid(np.matmul(test, np.transpose(W1)))
sgd_predictions =np.sign(np.matmul(predict_hidden,np.transpose(w1)))
print("   ")
print("Predictions_test_label=",sgd_predictions)
correct = 0
for i in range(len(testlabels)):
    if testlabels[i] == sgd_predictions[i]:
        correct += 1
    accuracy=correct / float(len(testlabels)) * 100.0
print("   ")
print("Accuracy of SGD=",accuracy)


# In[12]:


OUT= open("sgd_predictions", "w") 
content = str(sgd_predictions) 
OUT.write(content) 
OUT.close() 


# # Convolutional Neural Network

# In[13]:


print('#######################################################################')
print("Convolutional_Neural_Network")
print("       ")
print('#######################################################################')


# In[14]:


traindir = sys.argv[3]
testdir = sys.argv[4]


# In[ ]:


df = pd.read_csv(traindir+'/data.csv')
names = df['Name'].values
labels = df['Label'].values
traindata = np.empty((len(labels),3,3), dtype=np.float32)
for i in range(0,len(labels)):
    image_matrix = np.loadtxt(traindir+'/'+names[i])
    traindata[i] = image_matrix


# In[ ]:


df = pd.read_csv(testdir+'/data.csv')
names2 = df['Name'].values
labels2 = df['Label'].values 
testdata = np.empty((len(labels2),3,3),dtype=np.float32)
for i in range(0,len(labels2)):
    image_matrix = np.loadtxt(testdir+'/'+names2[i])
    testdata[i] = image_matrix


# In[ ]:


##Getting the value of output(before applying activation) by applying 2*2 filter and stride=1
def conv2(X, k):
    input_row, input_col = X.shape
    filter_row, filter_col = k.shape
    final_row, final_col = input_row - filter_row + 1, input_col - filter_col + 1
    final = np.empty((final_row, final_col))
    for x in range(final_row):
        for y in range(final_col):
            sub = X[x : x + filter_row, y : y + filter_col]
            final[x,y] = np.sum(sub * k)
    return final



# In[ ]:




# In[ ]:


def convnet(traindata,labels):
    c = np.random.rand(2,2)##filter
    print("initial_weight_c=",c)
    ##updating filter weight by calculating gradient
    obj=0
    for i in range(0,len(labels)):
        hidden_layer = conv2(traindata[i],c)
        for j in range(0,2,1):
            for k in range(0,2,1):
                hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
        output_layer = (hidden_layer[0][0] + hidden_layer[0][1]+hidden_layer[1][0]+hidden_layer[1][1])/4
        print('\n'+"initial_output_layer_train_data[i]=", output_layer)
        obj += (output_layer - labels[i])**2
        
    ep=0
    prevobj = np.inf
    stop=0.0001
    epochs = 1000
    eta = .1
    
    while(prevobj - obj >stop and i<epochs):
        prevobj = obj
        delta_c1 = 0
        delta_c2 = 0
        delta_c3 = 0
        delta_c4 = 0
#        output= (output_layer)**0.5
        for i in range(0,len(labels)):
            hidden_layer = conv2(traindata[i],c)
            for j in range(0,2,1):
                for k in range(0,2,1):
                    hidden_layer[j][k]= sigmoid(hidden_layer[j][k])
            out = (hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1])/4 - labels[i]
            ### Start calculating all the gradient value
            ###First gradient 
            delta_z1_delta_c1 = hidden_layer[0][0] *(1 - hidden_layer[0][0])*traindata[i][0][0]
            delta_z2_delta_c1 = hidden_layer[0][1] *(1 - hidden_layer[0][1])*traindata[i][0][1]
            delta_z3_delta_c1 = hidden_layer[1][0] *(1 - hidden_layer[1][0])*traindata[i][1][0]
            delta_z4_delta_c1 = hidden_layer[1][1] *(1 - hidden_layer[1][1])*traindata[i][1][1]
            delta_c1 += (out * (delta_z1_delta_c1 + delta_z2_delta_c1 + delta_z3_delta_c1 +delta_z4_delta_c1))/2
            ###2nd gradient  
            delta_z1_delta_c2 = hidden_layer[0][0] *(1 - hidden_layer[0][0])*traindata[i][0][1]
            delta_z2_delta_c2 = hidden_layer[0][1] *(1 - hidden_layer[0][1])*traindata[i][0][2]
            delta_z3_delta_c2 = hidden_layer[1][0] *(1 - hidden_layer[1][0])*traindata[i][1][1]
            delta_z4_delta_c2 = hidden_layer[1][1] *(1 - hidden_layer[1][1])*traindata[i][1][2]
            delta_c2 += (out *(delta_z1_delta_c2 + delta_z2_delta_c2 + delta_z3_delta_c2 +delta_z4_delta_c2))/2
            ###3rd gradient
            delta_z1_delta_c3 = hidden_layer[0][0] *(1 - hidden_layer[0][0])*traindata[i][1][0]
            delta_z2_delta_c3 = hidden_layer[0][1] *(1 - hidden_layer[0][1])*traindata[i][1][1]
            delta_z3_delta_c3 = hidden_layer[1][0] *(1 - hidden_layer[1][0])*traindata[i][2][0]
            delta_z4_delta_c3 = hidden_layer[1][1] *(1 - hidden_layer[1][1])*traindata[i][2][1]
            delta_c3 += (out *(delta_z1_delta_c3 + delta_z2_delta_c3 + delta_z3_delta_c3 +delta_z4_delta_c3))/2
            ###4th gradient
            delta_z1_delta_c4 = hidden_layer[0][0] *(1 - hidden_layer[0][0])*traindata[i][1][1]
            delta_z2_delta_c4 = hidden_layer[0][1] *(1 - hidden_layer[0][1])*traindata[i][1][2]
            delta_z3_delta_c4 = hidden_layer[1][0] *(1 - hidden_layer[1][0])*traindata[i][2][1]
            delta_z4_delta_c4 = hidden_layer[1][1] *(1 - hidden_layer[1][1])*traindata[i][2][2]
            delta_c4 += (out * (delta_z1_delta_c4 + delta_z2_delta_c4 + delta_z3_delta_c4 +delta_z4_delta_c4))/2            
        ###Updating filter weight  
        c[0][0] -= eta*delta_c1
        c[0][1] -= eta*delta_c2
        c[1][0] -= eta*delta_c3
        c[1][1] -= eta*delta_c4
        obj = 0 
        for i in range(0,len(labels)):
            hidden_layer = conv2(traindata[i],c)
            for j in range(0,2,1):
                for k in range(0,2,1):
                    hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
            output_layer = (hidden_layer[0][0] + hidden_layer[0][1]+hidden_layer[1][0]+hidden_layer[1][1])/4
            obj += (output_layer - labels[i])**2
            print('\n'+"Objective=",obj)
        ep=ep+1
    return c    


# In[ ]:


c=convnet(traindata,labels)
print('\n'+"Final_weight_c=",c)
print('\n'+"output_testlabel=")
convnet_predictions=[]
for i in range(0,len(labels2)):
    hidden_layer = conv2(testdata[i],c)
    list=[]
    for j in range(0,2,1):
        for k in range(0,2,1):
            hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
    output_layer = (hidden_layer[0][0] + hidden_layer[0][1]+hidden_layer[1][0]+hidden_layer[1][1])/4
    list.append(output_layer)
    predictions=[]
    for i in range(len(list)):
        if (output_layer < 0.5):
            convnet_predictions.append(-1)
            print(-1)
        else:
            convnet_predictions.append(1)
            print(1)


# In[ ]:


OUT= open("convnet_output", "w") 
content = str(convnet_predictions) 
OUT.write(content) 
OUT.close() 


# In[ ]:





# In[ ]:



