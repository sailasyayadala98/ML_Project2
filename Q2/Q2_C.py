import numpy as np
import math
import matplotlib.pyplot as plt
import Q2_AB as util1

def main():
    
    print('START Q2_C\n')
	#path_train=r'datasets/Q1_B_train.txt'
    path_train=r'datasets/Q1_B_train.txt'
    data=util1.read_data(path_train)
    data=data[data[:,0].argsort()]
    x=data[:,0]
    x=x.reshape(x.shape[0],1)
    y=data[:,1]
    y.reshape(x.shape[0],1)
    
    path_test=r'datasets/Q1_C_test.txt'
    data_test=util1.read_data(path_test)
    
    x_test=data_test[:,0]
    x_test=x_test.reshape(x_test.shape[0],1)
    y_test=data_test[:,1]
    y_test.reshape(x_test.shape[0],1)
    
    g=0.204
    
    y_pred=[]
    for point in x:
        y_pred.append(util1.local_weighted_linear_regression(point[0], x, y,g)[0,0])
    y_pred=np.array(y_pred)
    
    y_pred_test=[]
    for point in x_test:
        y_pred_test.append(util1.local_weighted_linear_regression(point[0], x, y,g)[0,0])
    
    y_pred_test=np.array(y_pred_test)
    
    print('MAE on Train data is',np.mean(np.abs(y-y_pred)))
    print('MAE on test data is',np.mean(np.abs(y_test-y_pred_test)))
    print('END Q2_C\n')

if __name__ == "__main__":
    main()
    