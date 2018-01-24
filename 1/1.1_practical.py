import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_xy(path): #Returns datasets x and y where x has column of 1.0 added
    raw = np.genfromtxt(path,delimiter=',')
    ones = np.ones((len(raw),1))
    data = np.append(ones,raw,axis=1)
    return data[:,:-1],data[:,-1]

def OLS(x,y): #Calculate Ordinary Least Square
    xt = x.transpose()
    r1 = np.dot(xt,x)
    r2 = np.linalg.inv(r1)
    r3 = np.dot(r2,xt)
    w = np.dot(r3,y)
    return w

def Emse(x,y,w): #Calculates Model Error
    Xw = np.dot(x,w)
    e = np.square(Xw - y).mean()
    return e

def plot(x,y,w,e,n): #Plots the data, e is error and n is name
    plt.scatter(x[:,1],y)
    fx = np.arange(0,1,0.01)
    fy = w[0] + w[1]*fx
    plt.plot(fx,fy,"r",lw=2.0)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    t = "OLS for " + n +" - Emse : {0:.4f}".format(e)
    plt.title(t)
    plt.grid()
    plt.axis('tight')
    plt.savefig("regression_plot_"+n+".png")
    plt.show()

def main():
    folder_path = "datasets/datasets/regression/"

    print("-------- 2.1.1 ------------")
    train_path = "train_2d_reg_data.csv"
    x,y = get_xy(folder_path + train_path)

    w = OLS(x,y)
    print("Weights: {}".format(w))

    e = Emse(x,y,w)
    print("Emse Train: {}".format(e))
   
    test_path = "test_2d_reg_data.csv"
    xt,yt = get_xy(folder_path + test_path)
    e = Emse(xt,yt,w)
    print("Emse Test: {}".format(e))

    
    print("-------- 2.1.2 ------------")
    train_path = "train_1d_reg_data.csv"
    x,y = get_xy(folder_path + train_path)
    w = OLS(x,y)
    print("Weights: {}".format(w))

    e = Emse(x,y,w)
    print("Emse Train: {}".format(e))
    plot(x,y,w,e,"train")

    test_path = "test_1d_reg_data.csv"
    xt,yt = get_xy(folder_path + test_path)

    e = Emse(xt,yt,w)
    print("Emse Test: {}".format(e))
    plot(xt,yt,w,e,"test")

    






if __name__ == "__main__":
    main()

