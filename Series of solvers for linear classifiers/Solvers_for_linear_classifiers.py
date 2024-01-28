# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:49:18 2020

@author: abdullah
"""

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
# ###################
# ## Standardize the code here: divide each feature of each
# ## datapoint by the length of each column in the training data
# ## return [traindata, testdata]
# ###################
def standardize_data(traindata, testdata):
# =============================================================================

############# Standardize traindata ##############
    row_train = len(traindata)
    row_test = len(testdata)
    col_train = len(traindata[0])

    u_train = []

    std_train = traindata
    std_test = testdata

    for i in range(col_train-1):

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



###################
## Solver for least squares (linear regression)
## return [w, w0]
###################
def least_squares(traindata, trainlabels):

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


    #while(prevobj - obj > 0.000000001):
    #while(prevobj - obj > 0):
    while(prevobj - obj > 0.001):

        prevobj = obj

        #### Reset delf to 0 #######
        for j in range(0, cols, 1):
            delf[j] = 0

        #### Compute delf ########
        for i in range(0, rows, 1):

            if(trainlabels.get(i) != None):
                dp = dotproduct(w, traindata[i])
                for j in range(0, cols, 1):
                    delf[j] += (trainlabels.get(i) - dp)*traindata[i][j]


        ###### Update w #######
        for j in range(0, cols, 1):
            w[j] += eta * delf[j]

        error = 0
        ##### Comput error #####
        for i in range(0, rows, 1):
            if(trainlabels.get(i) != None):
            #if(trainlabels.get(i) != 0):
                error += (trainlabels.get(i) - dotproduct(w, traindata[i]))**2

        obj = error
        #print ("Objective is ", error)

    return [w]





# =============================================================================
# ###################
# ## Solver for regularized least squares (linear regression)
# ## return [w, w0]
# ###################
def least_squares_regularized(traindata, trainlabels):
#

####################
## Initialize w
####################
    rows = len(traindata)
    cols = len(traindata[0])
    w = []
    for i in range(0, cols, 1):
        w.append(random.uniform(-0.01, 0.01))

    wlen = 0
    for m in range(0, cols, 1):
        wlen += w[m]**2
    wlen = math.sqrt(wlen)

########################
## Gradient Descent
########################

#### Initializations

    lam = 0.01
    eta = 0.001


    delf = []
    for j in range(0, cols, 1):
        delf.append(0)

    prevobj = 1000000000
    obj = prevobj - 10

### Main Iteration Loop #########


    #while(prevobj - obj > 0.000000001):
    #while(prevobj - obj > 0):
    while(prevobj - obj > 0.001):

        prevobj = obj

        wlen = 0
        for m in range(0, cols, 1):
            wlen += w[m]**2
        wlen = math.sqrt(wlen)

        #### Reset delf to 0 #######
        for j in range(0, cols, 1):
            delf[j] = 0

        #### Compute delf ########
        for i in range(0, rows, 1):

            if(trainlabels.get(i) != None):
                dp = dotproduct(w, traindata[i])
                for j in range(0, cols, 1):
                    delf[j] += (trainlabels.get(i) - dp)*traindata[i][j] - (2*lam*wlen)


        ###### Update w #######
        for j in range(0, cols, 1):
            w[j] += eta * delf[j]

        wlen = 0
        for m in range(0, cols, 1):
            wlen += w[m]**2
        wlen = math.sqrt(wlen)

        error = 0
        ##### Comput error #####
        for i in range(0, rows, 1):
            if(trainlabels.get(i) != None):
            #if(trainlabels.get(i) != 0):
                error += (trainlabels.get(i) - dotproduct(w, traindata[i]))**2
        error = error + (lam*wlen**2)
        obj = error
        #print ("Objective is ", error)

    return [w]




# ###################
# ## Solver for hinge loss
# ## return [w, w0]
# ###################
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


    #while(abs(prevobj - obj) > 0.000000001):
    #while(abs(prevobj - obj) > 0):
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





#
#
# ###################
# ## Solver for regularized hinge loss
# ## return [w, w0]
# ###################
def hinge_loss_regularized(traindata, trainlabels):
#

####################
## Initialize w
####################
    rows = len(traindata)
    cols = len(traindata[0])
    w = []
    for i in range(0, cols, 1):
        w.append(random.uniform(-0.01, 0.01))

    wlen = 0
    for m in range(0, cols, 1):
        wlen += w[m]**2
    wlen = math.sqrt(wlen)

########################
## Gradient Descent
########################

#### Initializations

    eta = 0.001
    lam = 0.01

    delf = []
    for j in range(0, cols, 1):
        delf.append(0)

    prevobj = 1000000000
    obj = prevobj - 10

### Main Iteration Loop #########


    #while(abs(prevobj - obj) > 0.000000001):
    #while(abs(prevobj - obj) > 0):
    while(abs(prevobj - obj) > 0.001):

        prevobj = obj

        #### Reset delf to 0 #######
        for j in range(0, cols, 1):
            delf[j] = 0

        #### Compute delf ########



        for i in range(0, rows, 1):
            if(trainlabels.get(i) != None):
                #a = max(0, 1-(trainlabels.get(i) * dotproduct(w, traindata[i])))
                a = (trainlabels.get(i) * dotproduct(w, traindata[i]))
                for j in range(0, cols, 1):
                    if (a<1):
                        delf[j] += ((trainlabels.get(i)*traindata[i][j])*(-1)) # + (lam*2*wlen)
                    elif (a>=1):
                        delf[j] += 0 #(lam*2*wlen)


        ###### Update w #######
        for j in range(0, cols, 1):
                w[j] -= eta * delf[j]

        wlen = 0
        for m in range(0, cols, 1):
            wlen += w[m]**2
        wlen = math.sqrt(wlen)



        error = 0
        ##### Comput error #####
        for i in range(0, rows, 1):
            if(trainlabels.get(i) != None):
                value = 1 - (trainlabels.get(i) * (dotproduct(w, traindata[i])))
                error += max(0, value)


        error = error + (wlen**2)
        obj = error
        #print ("Objective is ", error)

    return [w]




# ###################
# ## Solver for logistic regression
# ## return [w, w0]
# ###################
def logistic_loss(traindata, trainlabels):

####################
## Initialize w
####################
    rows = len(traindata)
    cols = len(traindata[0])

# =============================================================================
#     for i in range(rows):
#         if(trainlabels[int(i)] == -1):
#             trainlabels[int(i)] = 0
# =============================================================================

    w = []
    for i in range(0, cols, 1):
        w.append(random.uniform(-0.01, 0.01))

########################
## Gradient Descent
########################

#### Initializations

    #eta = 0.001
    eta = 0.001


    delf = []
    for j in range(0, cols, 1):
        delf.append(0)

    prevobj = 1000000000
    obj = prevobj - 10

### Main Iteration Loop #########


    #while(prevobj - obj > 0.000000001):
    #while(prevobj - obj > 0):
    while(prevobj - obj > 0.001):
    #while(prevobj - obj > 0.0000001):

        prevobj = obj

        #### Reset delf to 0 #######
        for j in range(0, cols, 1):
            delf[j] = 0

        #### Compute delf ########
        for i in range(0, rows, 1):

            if(trainlabels.get(i) != None):
                dp = dotproduct(w, traindata[i])
                for j in range(0, cols, 1):
                    a = 1 / (1 + math.exp((-1) * trainlabels.get(i) * dp))
                    b = math.exp((-1) * trainlabels.get(i) * dp)
                    c = (-1)*trainlabels.get(i)*traindata[i][j]
                    delf[j] += a*b*c


        ###### Update w #######
        for j in range(0, cols, 1):
            w[j] -= eta * delf[j]

        error = 0
        ##### Comput error #####
        for i in range(0, rows, 1):
            if(trainlabels.get(i) != None):
                error += math.log(1 + math.exp((-1) * trainlabels.get(i) * dotproduct(w, traindata[i])))

        obj = error
        #print ("Objective is ", error)

    return [w]



#
#
# ###################
# ## Solver for adaptive learning rate hinge loss
# ## return [w, w0]
# ###################
def hinge_loss_adaptive_learningrate(traindata, trainlabels):

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

    #eta = 0.001


    delf = []
    for j in range(0, cols, 1):
        delf.append(0)

    prevobj = 1000000000
    obj = prevobj - 10

### Main Iteration Loop #########


    #while(abs(prevobj - obj) > 0.000000001):
    #while(abs(prevobj - obj) > 0):
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



#======================= Adaptive pseudocode ==================
        eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001 ]
        bestobj = 1000000000000
        for k in range(0, len(eta_list), 1):

            eta = eta_list[k]

            ##update w
            ##insert code here for w = w + eta*dellf
            for j in range(0, cols, 1):
                w[j] -= eta * delf[j]

            ##get new error
            err = 0
            for i in range(0, rows, 1):
                if(trainlabels.get(i) != None):
                    ##update error
                    ##insert code to update the loss (which we call error here)
                    val = 1 - (trainlabels.get(i) * (dotproduct(w, traindata[i])))
                    err += max(0, val)

            #obj = err

          ##update bestobj and best_eta
          ##insert code here
            if (err < bestobj):
                bestobj = err
                best_eta = eta

      ##remove the eta for the next
      ##insert code here for w = w - eta*dellf



#==================== Adaptive pseudocode end =========================


        ###### Update w #######
        for j in range(0, cols, 1):
                w[j] -= best_eta * delf[j]



        error = 0
        ##### Comput error #####
        for i in range(0, rows, 1):
            if(trainlabels.get(i) != None):
                value = 1 - (trainlabels.get(i) * (dotproduct(w, traindata[i])))
                error += max(0, value)



        obj = error
        #print ("Objective is ", error)

    return [w]



# =============================================================================

original_stdout = sys.stdout
#### MAIN #########

###################
#### Code to read train data and train labels
###################
###### Abdullah start ##############
train_datafile = sys.argv[1]
f = open(train_datafile)
traindata = []

trainlabels_list = []


#i = 0
l_train = f.readline()


while(l_train != ''):
    a = l_train.split()
    trainlabels_list.append(int(a[0]))
    l2_train = []
    for j in range(1, len(a), 1):
        l2_train.append(float(a[j]))
    l2_train.append(1)
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

######## Abdullah end ##############
###################
#### Code to test data and test labels
#### The test labels are to be used
#### only for evaluation and nowhere else.
#### When your project is being graded we
#### will use 0 label for all test points
###################

###### Abdullah start ##############
test_datafile = sys.argv[2]
f = open(test_datafile)
testdata = []

testlabels_list = []

#i = 0
l_test = f.readline()


while(l_test != ''):
    b = l_test.split()
    testlabels_list.append(int(b[0]))
    l2_test = []
    for j in range(1, len(b), 1):
        l2_test.append(float(b[j]))
    l2_test.append(1)
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

######## Abdullah end ##############
rows = len(testdata)
# =============================================================================
[traindata, testdata] = standardize_data(traindata, testdata)
#
# =============================================================================

for p in range(6):

    if (p==0):

        print ("Working on ...... least_squares_prediction")
        [w] = least_squares(traindata, trainlabels)
        OUT = open("least_squares_prediction.txt", "w")

        for i in range(0, rows, 1):


            dp = dotproduct(w, testdata[i])
            sys.stdout = OUT
            if (dp < 0):
                print("-1 ",i)

            else:
                print("1 ",i)


        sys.stdout = original_stdout





    elif (p==1):

        print ("Working on ...... reg_least_squares_prediction")
        [w] = least_squares_regularized(traindata, trainlabels)
        OUT = open("reg_least_squares_prediction.txt", "w")

        for i in range(0, rows, 1):


            dp = dotproduct(w, testdata[i])
            sys.stdout = OUT
            if (dp < 0):
                print("-1 ",i)

            else:
                print("1 ",i)


        sys.stdout = original_stdout



    elif (p==2):

        print ("Working on ...... logistic_prediction")
        [w] = logistic_loss(traindata, trainlabels)

        OUT = open("logistic_prediction.txt", "w")

        for i in range(0, rows, 1):


            dp = dotproduct(w, testdata[i])
            sys.stdout = OUT
            if (dp < 0.5):
                print("-1 ",i)

            else:
                print("1 ",i)


        sys.stdout = original_stdout



    elif (p==3):

        print ("Working on ...... adaptive_eta_hinge_prediction")
        [w] = hinge_loss_adaptive_learningrate(traindata, trainlabels)

        OUT = open("adaptive_eta_hinge_prediction.txt", "w")

        for i in range(0, rows, 1):


            dp = dotproduct(w, testdata[i])
            sys.stdout = OUT
            if (dp < 0):
                print("-1 ",i)

            else:
                print("1 ",i)


        sys.stdout = original_stdout





    elif (p==4):

        print ("Working on ...... hinge_prediction")
        [w] = hinge_loss(traindata, trainlabels)


        OUT = open("hinge_prediction.txt", "w")

        for i in range(0, rows, 1):


            dp = dotproduct(w, testdata[i])
            sys.stdout = OUT
            if (dp < 0):
                print("-1 ",i)

            else:
                print("1 ",i)


        sys.stdout = original_stdout





    elif (p==5):

        print ("Working on ...... regularized_hinge_prediction")
        [w] = hinge_loss_regularized(traindata, trainlabels)

        OUT = open("regularized_hinge_prediction.txt", "w")

        for i in range(0, rows, 1):


            dp = dotproduct(w, testdata[i])
            sys.stdout = OUT
            if (dp < 0):
                print("-1 ",i)

            else:
                print("1 ",i)


        sys.stdout = original_stdout
