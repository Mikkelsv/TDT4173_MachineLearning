
import matplotlib.pyplot as plt

import numpy as np



def get_xy(path): #Gets data from path and inserts column of ones in start of array
    raw = np.genfromtxt(path,delimiter=',')
    ones = np.ones((len(raw),1)) 
    data = np.append(ones,raw,axis=1)
    return data[:,:-1],data[:,-1]

def get_z(X,w): #Calculates Z
    wt = w.transpose()
    z = np.dot(wt,X)
    return z

def get_sigma(z): #Calculates sigma of z
    return np.exp(z) / (1 + np.exp(z))

def Ece(w,x,y): #Calculates Cross entropy error function, ece value
    s = np.full(len(w),0.0)
    for i in range(len(y)):
        sigma = get_sigma(get_z(x[i,:],w))
        s += (sigma - y[i])*x[i]
    return s

def get_next_w(w,n,x,y): #Calculates w(n+1), returns w and ece
    nece = n*Ece(w,x,y)
    w = np.subtract(w,nece)
    return w, nece

def find_w(w,n,x,y,K): #Calculates w after K iterations using get_next_w, returns w and ece set
    e = []
    for i in range(K):
        w,nece = get_next_w(w,n,x,y)
        e.append(nece)
    return w,e

def alter_x(x): #edit x  to format fit w0 + w1x1 + w2x2 + w3x1^2 + w4x2^2
   l = len(x)
   nx = np.ones(shape=(l,5))
   for i in range(l):
       nx[i,0:3] = x[i]
       nx[i,3]= x[i,1]**2
       nx[i,4]= x[i,2]**2
   return nx

    


def plot(x,y,w,linear=True):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    l = int(np.sum(y))
    cv = len(w)
    a = np.ones(shape=(l,cv))
    ac = 0
    b = np.ones(shape=(len(y)-l,cv))
    bc = 0
    for i in range(len(y)):
        if(y[i]>0.5):
            a[ac] = x[i]
            ac += 1
        else:
            b[bc] = x[i]
            bc += 1
    ax1.scatter(a[:,1],a[:,2],s=40,c="blue") 
    ax1.scatter(b[:,1],b[:,2],s=40,c="red",marker="s") 
    
    #Not linear assumes form w0 + w1x1 + w2x2 + w3x1^2 + w4x2^2
    if(linear):
        fx = np.arange(0,1,0.01)
        fy = -w[0]/w[2] - w[1]*fx/w[2]
        plt.plot(fx,fy,"r",lw=2.0)
    else:
        xs = np.linspace(0.0,1.0,100)
        ys = xs
        X,Y = np.meshgrid(xs,ys)
        F =  w[0] + w[1]*X + w[2]*Y +w[3]*X**2 + w[3]*Y**2 
        plt.contour(X,Y,F,[0])
    plt.axis('tight')
    plt.show()
    

def main():
    n = 0.1 #Learning rate
    K = 1000 #Iterations

    #Data paths
    folder_path = "datasets/datasets/classification/"

    print("------2.2.1-------")
    train_path = "cl_train_1.csv"
    x,y = get_xy(folder_path + train_path)
    w = np.full(3,1.0) 

    w,e = find_w(w,n,x,y,K)

    print("Weights test: {}".format(w))
    plt.plot(e)
    plt.show()
    plot(x,y,w,True)

    test_path = "cl_test_1.csv"
    x,y = get_xy(folder_path + test_path)
    w = np.full(3,1.0) 
    w,e = find_w(w,n,x,y,K)

    print("Weights test: {}".format(w))


        
if __name__ == "__main__":
    main()


