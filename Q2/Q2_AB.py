import numpy as np
import math
import matplotlib.pyplot as plt

def read_data(path):
    dataset = np.genfromtxt(path,delimiter=' ')
    dataset = dataset[:,[1,3]]
    return dataset

def weights(point,X,g=0.204):
    m,n = np.shape(X)
    weights=np.mat(np.eye((m)))
    
    for j in range(m):
        diff=point-X[j]
        
        weights[j,j]=np.exp(np.dot(diff,diff.T)/-(2*g**2))
    
    return weights

def local_weighted_params(point,X,y,g=0.204):
    wt=weights(point,X,g)
    y=y.reshape(X.shape[0],1)
    params=(X.T*(wt*X)).I*(X.T*(wt*y))
    return params

def local_weighted_linear_regression(point,X,y,g=0.204):
    m=X.shape[0]
    #X=X.reshape(m,1)
    X = np.append(np.ones(m).reshape(m,1),X,axis=1)
    point=np.array([1,point])
    #y=y.reshape(m,1)
    params=local_weighted_params(point, X, y, g)
    predicted=np.dot(point,params)
    return predicted
        

def main():
	
    print('START Q2_AB\n')
    path_train=r'datasets/Q1_B_train.txt'
    data=read_data(path_train)
    data=data[data[:,0].argsort()]
    x=data[:,0]
    x=x.reshape(x.shape[0],1)
    y=data[:,1]
    y.reshape(x.shape[0],1)
    
    g=0.204
    y_pred=[]
    for point in x:
        y_pred.append(local_weighted_linear_regression(point[0], x, y,g)[0,0])
    
    y_pred=np.array(y_pred)
    plt.scatter(x,y, color='orange',marker='o')
    plt.plot(x,y_pred,color='black')
    plt.title('Locally Weighted Linear Regression')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    print('END Q2_AB\n')
if __name__ == "__main__":
    main()
