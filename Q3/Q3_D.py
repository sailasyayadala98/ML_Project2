import numpy as np
import matplotlib.pyplot as plt
import math
import Q3_AB as logreg
import Q3_C as loov

def main():
	#print('START Q3_C\n')
    print('START Q3_D\n')
    path=r'datasets/Q3_data.txt'
    data=logreg.read_data(path)
    data=np.array(data)
    #Changing dataypes
    X=data[:,[0,1]]
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
    
    print('Leave one out validation score with age column is ',loov.validation(X, y, init_parameters))
    
	#print('END Q3_C\n')
    print('END Q3_D\n')


if __name__ == "__main__":
    main()
    