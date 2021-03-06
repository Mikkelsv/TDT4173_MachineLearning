
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

def plot(x,y,w,linear,name):
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
    t = name + " plot\n"
    
    #Not linear assumes form w0 + w1x1 + w2x2 + w3x1^2 + w4x2^2
    if(linear):
        fx = np.arange(0,1,0.01)
        fy = -w[0]/w[2] - w[1]*fx/w[2]
        plt.plot(fx,fy,"r",lw=2.0)
        t += "Formula: w0 + w1x1 + w2x2\nWeights: {0:.4f}, {1:.4f}, {2:.4f}".format(w[0], w[1],w[2])
    else:
        xs = np.linspace(0.0,1.0,100)
        ys = xs
        X,Y = np.meshgrid(xs,ys)
        F =  w[0] + w[1]*X + w[2]*Y +w[3]*X**2 + w[3]*Y**2 
        plt.contour(X,Y,F,[0])
        t += "Formula: w0 + w1x1 + w2x2 + w3(x1)^2 + w4(x2)^2\n Weights: {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}".format(w[0],w[1],w[2],w[3],w[4])
    plt.title(t) 
    plt.axis('tight')
    plt.xlabel("X1-axis")
    plt.ylabel("X2-axis")
    plt.savefig("classification_"+name+"_plot.png")
    plt.clf()

def plot_e(e,name,t="Plot"):
    e = np.array(e)
    plt.clf()
    c = ['b','g','r','c','m']
    l = len(e[0]) 
    for i in range(l):
        plt.plot(e[:,i],c[i]+'o',label='w'+str(i))

    t = name + " Ece development \nFormula: w0 + w1x1 + w2x2"
    if(l==5):
        t += " + w3(x1)^2 + w4(x2)^2"
    plt.title(t)
    plt.yscale('log')
    plt.xlabel("Iterations")
    plt.ylabel("Ece value")
    plt.legend()
    plt.savefig("classification_error_"+name+"_plot.png")


def main():
    n = 0.1 #Learning rate
    K = 1000 #Iterations

    #Data paths
    folder_path = "datasets/datasets/classification/"

    print("------2.2.1-------")
    #-----2.1 Train data----
    train_path = "cl_train_1.csv"
    x,y = get_xy(folder_path + train_path)
    w = np.full(3,1.0) 
    w,e = find_w(w,n,x,y,K)

    print("Weights training: {}".format(w))
    plot_e(e,"2_1_train")
    plot(x,y,w,True,"2_1_train")

    #-----2.1 Test data----
    test_path = "cl_test_1.csv"
    xt,yt = get_xy(folder_path + test_path)
    wt = np.full(3,1.0) 
    wt,et = find_w(wt,n,xt,yt,K)

    print("Weights test: {}".format(wt))
    plot_e(et,"2_1_test")
    plot(xt,yt,w,True,"2_1_test")

    print("------2.2.2-------")
    train_path = "cl_train_2.csv"


    #------2.2 Train data----------
    x,y = get_xy(folder_path + train_path)
    x = alter_x(x)
    w = np.full(5,1.0) 
    w,e = find_w(w,n,x,y,K)

    print("Weights train: {}".format(w))
    plot(x,y,w,False,"2_2_train")
    plot_e(e,"2_2_train") 

    #------2.2 Test data----------
    xt,yt = get_xy(folder_path + train_path)
    xt = alter_x(xt)
    plot(xt,yt,w,False,"2_2_test")

    #------2.2 Linear Plot--------
    xl,yl = get_xy(folder_path + train_path)
    wl = np.full(3,1.0) 
    wl,el = find_w(wl,n,xl,yl,K)
    print("Weights train for linear plot: {}".format(wl))
    plot(xl,yl,wl,True,"2_2_linear")

        
if __name__ == "__main__":
    main()


