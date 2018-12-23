import numpy as np
from numpy import append, array, int8, uint8, zeros
from array import array as pyarray
import math
import pandas as pd
import os,struct
from time import time
inputNumber = 784
hiddenNumber = 25 #隐层数量为100
outputNumber = 10
eta = 0.2 #学习率20%
debug = True

def load_mnist(image_file, label_file, path="."):
    digits=np.arange(10)

    fname_image = os.path.join(path, image_file)
    fname_label = os.path.join(path, label_file)

    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows*cols), dtype=uint8)
    labels = zeros(N, dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((1, rows*cols))
        labels[i] = lbl[ind[i]]
    #images = [im/255.0 for im in images]
    images = [im/255.0 - 0.5 for im in images]

    return images, labels

def func(x):
    return 1/(1+math.exp(-x))

def BP(train_x,train_y):
    '''输入为784维归一化后的向量(train_x单个元素必须为784维向量)，与标签，输出神经网络的权重参数'''
    #进行循环，循环次数视误差而定
    #每次循环的时候，遍历所有训练集，进行一次前向和后向
    X = np.zeros(inputNumber) #输入层
    Y = np.zeros(hiddenNumber) #隐层
    O = np.zeros(outputNumber) #输出层
    d = np.zeros(outputNumber) #输入参考
    W = np.zeros((hiddenNumber,outputNumber)) #隐层到输出层的权重
    V = np.zeros((inputNumber,hiddenNumber)) #输入层到隐层的权重
    delta = np.zeros(outputNumber) #输出层反传误差
    loopNum = 10
    currentNum = 0
    for loop in range(loopNum):
        for tx,ty in zip(train_x,train_y):
            print(currentNum)
            currentNum += 1
            if currentNum > 500 and debug: #测试用,python好慢
                break

            X = np.array(tx)
            #重新对准确信息赋值
            for k in range(outputNumber):
                d[k] = 0.0
            d[train_y] = 1.0

            for j in range(hiddenNumber):
                Y[j] = 0
                #net_j
                for i in range(inputNumber):
                    Y[j] += X[i] * V[i][j]
                Y[j] = func(Y[j])
            
            for k in range(outputNumber):
                O[k] = 0
                #net_k
                for j in range(hiddenNumber):
                    O[k] += Y[j]*W[j][k]
                O[k] = func(O[k])
                delta[k] = (d[k] - O[k])*O[k]*(1-O[k])

            for j in range(hiddenNumber):
                for k in range(outputNumber):
                    W[j][k] += eta*delta[k]*Y[j]
            
            for j in range(hiddenNumber):
                sum = 0
                for k in range(outputNumber):
                    sum += delta[k]*W[j][k]
                for i in range(inputNumber):
                    V[i][j] += eta*sum*Y[j]*(1-Y[j])*X[i]

    return (V,W)

def test(test_x,test_y,V,W):
    '''测试给出的神经网络的准确度'''
    Y = np.zeros(hiddenNumber) #隐层
    O = np.zeros(outputNumber) #输出层
    num = 0
    ac_num = 0
    for tx,ty in zip(test_x,test_y):
        num += 1
        if num > 100 and debug:
            break
        X = np.array(tx)
        for j in range(hiddenNumber):
            Y[j] = 0
            #net_j
            for i in range(inputNumber):
                Y[j] += X[i] * V[i][j]
            Y[j] = func(Y[j])

        maxP = 0
        maxNum = 0
        for k in range(outputNumber):
            O[k] = 0
            #net_k
            for j in range(hiddenNumber):
                O[k] += Y[j]*W[j][k]
            O[k] = func(O[k])
            if O[k]>maxP:
                maxP = O[k]
                maxNum = k

        if k == ty:
            ac_num += 1
    return ac_num / num
        




if __name__ == "__main__":
    train_image, train_label = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    test_image, test_label = load_mnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    print("读取数据集完毕")
    t1 = time()
    V,W = BP(train_image,train_label)
    t2 = time()
    print("训练时间",t2-t1)
    ac = test(test_image,test_label,V,W)
    print("测试准确率：",ac)