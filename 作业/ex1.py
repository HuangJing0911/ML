import numpy as np
from prml.preprocess import PolynomialFeature
from prml.linear import (
    LinearRegression,
    RidgeRegression,
    BayesianRegression
)

class polynomial_model:
    '''f = kx + b'''
    model_order = 1                 # 默认最高阶数为1，即y = kx+b 
    loss_function = "Square_loss"   # 默认损失函数为平方损失函数
    learning_rate = 0.05            # 默认学习率为0.05
    epoch = 1000                    # 训练轮次默认为1000轮
    x = []                          # 输入数据x
    y = []                          # 输入数据y
    poly_k = []                     # 最终训练的结果，即每次幂的系数
    b = 0                           # 最终训练结果的偏置项                
    
    # 初始化模型
    def init_model(self, model_order, loss_function, learning_rate, epoch):
        self.model_order = model_order
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.poly_k = np.zeros(shape=(1, model_order))
        print("Successfully init the model!")

    # 读取数据
    def data_read(self, filepath):  
        f = open(filepath,'r')
        lines = f.readlines()
        for line in lines:
            data = line.strip('\n').split('\t')
            self.x.append(float(data[0]))
            self.y.append(float(data[1]))
        print('Successfully loading data!')
        print('Length of data is:', len(self.x))
    
    # 训练模型
    def train(self):
        '''略'''
        print("Starting Training...............")
        print("Sucessfully Finish Training!")
    
    

# 类的使用示例
if __name__ == '__main__':
    # 测试模型初始化
    a = polynomial_model()
    a.init_model(2, 'Square_loss', 0.1, 1500)
    print("The order of model is:", a.model_order)
    print("The loss function of model is:", a.loss_function)
    print("The learning rate of model is:", a.learning_rate)
    print("The epochs of model is:", a.epoch)
    print("The poly_k of model is:", a.poly_k, '\n')
    # 测试模型数据读入
    a.data_read('C:/Users/96342/Desktop/ex1_data.txt')
    print("The value of x is:", a.x)
    print("The value of y is:", a.y, '\n')
    # 测试模型训练
    a.train()