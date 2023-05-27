import numpy as np
import matplotlib.pyplot as plt
import math

def read_data(path):
    f=open(path)
    td=[]
    for lines in f.readlines():
        line = lines.strip().split(',')
        dat=[]
        for l in line:
            l=l.replace(')','').replace('(','').replace(' ','')
            dat.append(l)
        td.append(dat)
    return td

def activation(input):    
    output = 1 / (1 + np.exp(-input))
    return output

def optimize(x, y,learning_rate,iterations,parameters): 
    size = x.shape[0]
    weight = parameters["weight"] 
    bias = parameters["bias"]
    for i in range(iterations): 
        sigma = activation(np.dot(x, weight) + bias)
        loss = -1/size * np.sum(y * np.log(sigma)) + (1 - y) * np.log(1-sigma)
        dW = 1/size * np.dot(x.T, (sigma - y))
        db = 1/size * np.sum(sigma - y)
        weight -= learning_rate * dW
        bias -= learning_rate * db 
    
    parameters["weight"] = weight
    parameters["bias"] = bias
    return parameters

def train(x, y, learning_rate,iterations,parameter):
    parameters_out = optimize(x, y, learning_rate, iterations ,parameter)
    return parameters_out

def plot_3D(X,y,parameters,n_iter):
    z = lambda x,Y: (-parameters['bias']-parameters['weight'][0]*x-parameters['weight'][1]*Y) / parameters['weight'][2]
    
    mesh=np.linspace(-1.5,1.5,80)
    x,Y=np.meshgrid(mesh,mesh)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x,Y,z(x,Y))
    ax.plot3D(X[y==0,0],X[y==0,1],X[y==0,2],'xb')
    ax.plot3D(X[y==1,0],X[y==1,1],X[y==1,2],'og')
    ax.view_init(60,30)
    title='Logistic Regression Hyperplane with iterations: ' + str(n_iter)
    
    plt.title(title)
    plt.show()
    

def predictions(x,parameters):
    z=np.dot(x,parameters['weight'])+parameters['bias']
    pred=[]
    for i in activation(z):
        
        if i>=0.5:
            pred.append(1)
        else:
            pred.append(0)
    pred=np.array(pred)
    return pred

def accuracy(y,y_pred):
    right = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            right += 1
    return right / float(len(y))



def main():
    
    print('START Q3_AB\n')
    path=r'datasets/Q3_data.txt'
    data=read_data(path)
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
    iteration=[10,20,50,100,150]
    
    for n_iter in iteration:
        parameters=train(X,y,0.01,n_iter,init_parameters)
        y_pred = predictions(X, parameters)
        print('Accuracy with no of iterations =',str(n_iter),'is ',str(accuracy(y, y_pred)))
        plot_3D(X,y,parameters,n_iter)
    
    print('END Q3_AB\n')
    
    
    
    
    
    
    
        


if __name__ == "__main__":
    main()
