import tensorflow as tf
import matplotlib.pyplot as plt
from ex1 import polynomial_model
import random

model = polynomial_model()
# 读取MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 获得数据集相关参数
train_num = mnist.train.num_examples
validation_num = mnist.validation.num_examples
test_num = mnist.test.num_examples
print('MNIST数据集的个数')
print(' >>>train_nums=%d' % train_num,'\n',
      '>>>validation_nums=%d'% validation_num,'\n',
      '>>>test_nums=%d' % test_num,'\n')

# 获得数据值
train_data = mnist.train.images   #所有训练数据
val_data = mnist.validation.images  #(5000,784)
test_data = mnist.test.images       #(10000,784)
print('>>>训练集数据大小：',train_data.shape,'\n',
      '>>>一副图像的大小：',train_data[0].shape)

# 获得标签值
train_labels = mnist.train.labels     #(55000,10)
val_labels = mnist.validation.labels  #(5000,10)
test_labels = mnist.test.labels       #(10000,10)

# 训练并打印图像
plt.figure()
for i in range(10):
    im = train_data[i].reshape(28,28)
    model.train()
    k = random.randrange(9)
    plt.title("random test" + str(k), fontsize=20)
    plt.imshow(im,'gray')
    plt.pause(0.8)
plt.show()