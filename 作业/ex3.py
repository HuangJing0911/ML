import tensorflow as tf
import matplotlib.pyplot as plt
from ex1 import polynomial_model
import random
import numpy as np
from numpy.linalg import inv
import xlrd
from prml.preprocess import PolynomialFeature
from prml.linear import (
    LinearRegression,
    RidgeRegression,
    BayesianRegression
)

def create_toy_data(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t

def func(x):
    return np.sin(2 * np.pi * x)


# ex3_1:Regularization Path 曲线
def Regularization_Path():
    x_train, y_train = create_toy_data(func, 10, 0.25)
    x_test = np.linspace(0, 1, 100)
    y_test = func(x_test)
    alpha = 1
    degree = 3
    feature = PolynomialFeature(degree=degree)
    X_train = feature.transform(x_train)
    X_test = feature.transform(x_test)
    # print(X_train,x_train)

    # 计算没有正则化时的w=(X^TX)^-1X^TY及其2-范数
    w_infinite = inv(np.dot(np.transpose(X_train),X_train))
    w_infinite = np.dot(w_infinite,np.transpose(X_train))
    w_infinite = np.dot(w_infinite,y_train)
    w_norm = np.linalg.norm(w_infinite, ord=2)
    print(w_infinite)
    print(w_norm)

    # 求解在lamda值不同的情况下w的值
    w = []
    rate = []               # rate为横坐标  
    learning_time = 100     # lamda的变化次数，即坐标个数
    for i in range(learning_time):
        model = RidgeRegression(alpha=alpha)
        model.fit(X_train, y_train)
        y = model.predict(X_test)
        error = np.mean(np.square(y - y_test))
        # print("alpha=" + str(round(alpha,6)) + ":mse=" + str(round(error,6)))
        # print(model.w)
        # 求解横坐标，即矩阵的范数
        w.append(model.w)
        lamda_norm = np.linalg.norm(model.w, 2)
        rate.append(lamda_norm / w_norm)
        alpha = alpha * 0.75
    w = np.array(w)         # w为100*4的矩阵
    rate = np.array(rate)   # rate为100*1的列向量
    # print(rate)

    # 画图
    for i in range(degree+1):
        w_lamda = w[:,i]
        plt.scatter(rate, w_lamda)
        plt.plot(rate, w_lamda, label="w"+str(i)+"("+str(round(w_infinite[i],3))+")")
    plt.title("Regularization Path")
    plt.xlabel("Rate of norm")
    plt.ylabel("W with lamda")
    plt.legend()
    plt.show()



# ex3_2: 拟合图片中的点
def Estimate_Line():
    xl = xlrd.open_workbook("E:\Study\研一\学习\机器学习\作业\附件：第三次作业_第二题.xlsx")
    data = xl.sheets()[1]   # 选择第二个表格作为读入的data
    rows = data.nrows   # data共116行,3列
    x_train = []
    y_train = []
    # 把表格中的数据变为numpy数组
    for i in range(1,rows):
        x = data.cell(i,2).value * -1
        y = data.cell(i,1).value * -1
        x_train.append(x)
        y_train.append(y)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test= x_train
    # 训练模型
    alpha = 0.01
    feature = PolynomialFeature(degree=20)
    X_train = feature.transform(x_train)
    X_test = feature.transform(x_test)
    model = RidgeRegression(alpha=alpha)
    model.fit(X_train, y_train)
    y = model.predict(X_test)
    # 画图
    plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
    # plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
    plt.plot(x_test, y, c="r", label="fitting")
    plt.title("Fitting the point on pic")
    plt.show()

if __name__ == '__main__':
    Regularization_Path()
    Estimate_Line()