from __future__ import print_function
import numpy as np

def readdata():


    X_train = np.ones([60000,28,28])
    y_train = np.ones(60000,int)
    X_test = np.ones([10000,28,28])
    y_test = np.ones(10000,int)

    data_2class = np.loadtxt('/Users/wangruqin/VScode/kadai1/spiral_data/2class.txt',dtype=np.float32)
    for i in range (60000):
        X_train[i][0][0] = data_2class[i%1430][0]
        X_train[i][0][1] = data_2class[i%1430][1]
        y_train[i] = data_2class[i%1430][2]
    
    for j in range (10000):
        X_test[j][0][0] = data_2class[1430+j%286][0]
        X_test[j][0][1] = data_2class[1430+j%286][1]
        y_test[j] = data_2class[1430+j%286][2]

    # y_train = y_train.astype(int)



    return X_train, y_train, X_test, y_test

readdata()

