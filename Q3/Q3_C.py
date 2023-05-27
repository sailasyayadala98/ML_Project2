import numpy as np
import matplotlib.pyplot as plt
import math
import Q3_AB as logreg

def validation(X,y,init_parameters,learning_rate=0.01,n_iter=50):
    right_pred=0
    for i in range(X.shape[0]):
        row=X[i,:]
        real_label=y[i]
        
        X_new=np.delete(X,i,axis=0)
        y_new = np.delete(y,i)
        row=row.reshape(1,X.shape[1])
        
        parameters=logreg.train(X_new,y_new,0.01,n_iter,init_parameters)
        predicted_label = logreg.predictions(row, parameters)
        
        if real_label==predicted_label[0]:
            right_pred+=1
        
    return right_pred/X.shape[0]

def main():
    print('START Q3_C\n')
    path=r'datasets/Q3_data.txt'
    data=logreg.read_data(path)
    data=np.array(data)
    #Changing dataypes
    X=data[:,:-1]
    X=X.astype(np.float64)
    y=data[:,-1]
    y[y=='W']=0
    y[y=='M']=1
    y=y.astype(np.float64)
    #Standarization
    X=(X-X.mean(axis=0))/X.std(axis=0)
    
    #initialize parameters
    init_parameters = {} 
    init_parameters["weight"] = np.zeros(X.shape[1])
    init_parameters["bias"] = 0
    learning_rate=0.01
    
    print('Leave one out validation score with age column is ',validation(X, y, init_parameters))
    
	#print('END Q3_C\n')
    print('END Q3_C\n')
    


if __name__ == "__main__":
    main()
    