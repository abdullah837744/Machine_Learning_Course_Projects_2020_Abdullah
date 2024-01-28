
"""
Created on Wed Nov 11 9:38:12 2020

@author: abdulah
"""
#####################################

import sys
import random
import math

#### FUNCTIONS #########

###################
#### Supplementary function to compute dot product
###################
def dotproduct(u, v):
	assert len(u) == len(v), "dotproduct: u and v must be of same length"
	dp = 0
	for i in range(0, len(u), 1):
		dp += u[i]*v[i]
	return dp

# =============================================================================
# ## Standardization Function
#==========================================================================
def standardize_data(traindata, testdata):
# =============================================================================
    row_train = len(traindata)
    row_test = len(testdata)
    col_train = len(traindata[0])
    
    u_train = []
    
    std_train = traindata
    std_test = testdata
    
    for i in range(col_train):
        
        t = 0 
        
        for j in range(row_train):
            
            t = t + (traindata[j][i])**2
        
        root_t = math.sqrt(t)
        u_train.append(root_t)
        
        if (root_t != 0):
            
            for k in range(row_train):           
                std_train[k][i] = traindata[k][i] / u_train[i]
                               
            for m in range(row_test):
                std_test[m][i] = testdata[m][i] / u_train[i]
            
   
    return [std_train, std_test]
    
#===============================================================
# ========== Hinge Loss function =================================
#===============================================================
def hinge_loss(traindata, trainlabels):
    
####################
## Initialize w
####################
    rows = len(traindata)
    cols = len(traindata[0])
    w = []
    for i in range(0, cols, 1):
        w.append(random.uniform(-0.01, 0.01))
        
########################
## Gradient Descent
########################
        
#### Initializations
        
    eta = 0.001
    
    
    delf = []
    for j in range(0, cols, 1):
        delf.append(0)
        
    prevobj = 1000000000
    obj = prevobj - 10
    
### Main Iteration Loop #########
    
    

    while(abs(prevobj - obj) > 0.001):
        
        prevobj = obj
        
        #### Reset delf to 0 #######
        for j in range(0, cols, 1):
            delf[j] = 0
            
        #### Compute delf ########
        
        
        
        for i in range(0, rows, 1):            
            if(trainlabels.get(i) != None):
                a = max(0, 1-(trainlabels.get(i) * dotproduct(w, traindata[i])))
                for j in range(0, cols, 1):
                    if (a>0):
                        delf[j] += (trainlabels.get(i)*traindata[i][j])*(-1)
                    else:
                        delf[j] += 0
              
                    
        ###### Update w #######
        for j in range(0, cols, 1):
                w[j] -= eta * delf[j]

                 
            
        error = 0
        ##### Comput error #####
        for i in range(0, rows, 1):
            if(trainlabels.get(i) != None):
                value = 1 - (trainlabels.get(i) * (dotproduct(w, traindata[i])))
                error += max(0, value)
                
                
                
        obj = error
        #print ("Objective is ", error)
        
    return [w]

#### MAIN #########
original_stdout = sys.stdout
###################
#### Code to read train data and train labels
###################
###### Abdullah start ##############    
train_datafile = sys.argv[1]
f = open(train_datafile)
traindata = []

trainlabels_list = []



l_train = f.readline()


while(l_train != ''):
    a = l_train.split()
    trainlabels_list.append(int(a[0]))
    l2_train = []
    for j in range(1, len(a), 1):
        l2_train.append(float(a[j]))
    #l2_train.append(1)
    traindata.append(l2_train)
    l_train = f.readline()
    
    
rows_train = len(traindata)
cols_train = len(traindata[0])

trainlabels = {}
for i in range(rows_train):
    trainlabels[int(i)] = int(trainlabels_list[i])
    if(trainlabels[int(i)] == 0):
        trainlabels[int(i)] = -1


f.close()

#===============================================
#### Code to test data and test labels
#==============================================  
test_datafile = sys.argv[2]
f = open(test_datafile)
testdata = []

testlabels_list = []


l_test = f.readline()


while(l_test != ''):
    b = l_test.split()
    testlabels_list.append(int(b[0]))
    l2_test = []
    for j in range(1, len(b), 1):
        l2_test.append(float(b[j]))
    #l2_test.append(1)
    testdata.append(l2_test)
    l_test = f.readline()
    
    
rows_test = len(testdata)
cols_test = len(testdata[0])

testlabels = {}
for i in range(rows_test):
    testlabels[int(i)] = int(testlabels_list[i])
    if(testlabels[int(i)] == 0):
        testlabels[int(i)] = -1


f.close()

####################################################################
#=================================================================
#=============Feature learning ===================================
#=================================================================
#####################################################################
k = int(sys.argv[3])

newdata_train =[]
newdata_test=[]

Ztrain = []
Ztest = []

for m  in range(0,k,1):
    


    list_train=[]
    list_test=[]


    w=[]
    for j in range(0, cols_train, 1):
        w.append(random.uniform(1,-1))

        
    #======== creating w0 vector ====================
    w0 = []
    a = []
    for j in range(0,rows_train,1):
        b = dotproduct(w,traindata[j])
        a.append(b)
    
    w0 = random.uniform(min(a),max(a))
 
    
        
    
    #=================================================

    for i in range(0,rows_train):
        
        dp_train=dotproduct(w,traindata[i]) + w0
        sign_train=int(math.copysign(1, dp_train))
        val_train=int((1+sign_train)/2)
        list_train.append(val_train)
        
    for i in range(0,rows_test):
        
        dp_test=dotproduct(w,testdata[i]) + w0
        sign_test=int(math.copysign(1, dp_test))
        val_test=int((1+sign_test)/2)
        list_test.append(val_test)


    newdata_train.append(list_train)
    newdata_test.append(list_test)


#================= transposing to make it n x k ================================

for i in range(0,rows_train,1):
    Ztrain.append([row[i] for row in newdata_train])
    
for i in range(0,rows_test,1):
    Ztest.append([row[i] for row in newdata_test])

    
f.close()

#===================================================================
#====================================================================

#======== Absorbing w0 by appending an extra column of 1s ============

for i in range(rows_train):
    traindata[i].append(1)
    Ztrain[i].append(1)

for i in range(rows_test):
    testdata[i].append(1)
    Ztest[i].append(1)
    

# =============================================================================
# =========================Standardizing========================================
[traindata, testdata] = standardize_data(traindata, testdata)
[Ztrain, Ztest] = standardize_data(Ztrain, Ztest)
 
#======================================================================
# # ===================== Prediction ============================================
rows = len(testdata)
 
for p in range(2):
    
    if (p==0):
        
        print ("Working on ...... Original data")
        [wX] = hinge_loss(traindata, trainlabels)
        OUT = open("prediction_original_data.txt", "w")

        for i in range(0, rows, 1):
    

            dp = dotproduct(wX, testdata[i])
            sys.stdout = OUT
            if (dp < 0):
                #print("-1 ",i)
                print("-1 ")
        
            else:
                #print("1 ",i)
                print("1 ")
        

        sys.stdout = original_stdout         
        
        
        
        
    elif (p==1):
        

                
        print ("Working on ......New feature data")
        [wZ] = hinge_loss(Ztrain, trainlabels)
        OUT = open("prediction_new_feature_data.txt", "w")

        for i in range(0, rows, 1):
    

            dp = dotproduct(wZ, Ztest[i])
            sys.stdout = OUT
            if (dp < 0):
                #print("-1 ",i)
                print("-1 ")
        
            else:
                #print("1 ",i)
                print("1 ")
        

        sys.stdout = original_stdout
