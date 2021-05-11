import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier 
import matplotlib.pyplot as plt

# ex4_1:多项逻辑斯蒂回归的编程实现
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
batch_size=128
X = tf.placeholder(tf.float32,[None,784],name='X_placeholder')
Y = tf.placeholder(tf.int32, [None,10],name='Y_placehoder')

w = tf.Variable(tf.random_normal(shape=[784,10],stddev=0.01),name='weights')
b = tf.Variable(tf.zeros([1,10]),name='bias')

# W*x+b
logits=tf.matmul(X,w)+b

entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y,name='loss')
loss=tf.reduce_mean(entropy)

learning_rate=0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

n_epochs = 30
init=tf.global_variables_initializer()
with tf.Session() as sess:
    writer=tf.summary.FileWriter('./graphs/logistic_reg',sess.graph)

    start_time=time.time()
    sess.run(init)
    n_batches=int(mnist.train.num_examples/batch_size)

    # 训练模型
    for i in range(n_epochs):
        total_loss=0
        for _ in range(n_batches):
            X_batch, Y_batch =mnist.train.next_batch(batch_size)
            _,loss_batch =sess.run([optimizer,loss],feed_dict={X:X_batch,Y:Y_batch})
            total_loss +=loss_batch
        print ('Average loss epoch {0}:{1}'.format(i,total_loss/n_batches))
    print ('Total time: {0} seconds'.format(time.time()-start_time))
    print ('optimizatin Finished')


    preds = tf.nn.softmax(logits)                               # 得到每张图片对每个数字种类预测的概率
    correct_preds=tf.equal(tf.argmax(preds,1),tf.argmax(Y,1))   # 找到预测值preds和Y的每一行最大的下标并逐个判断
    accuracy=tf.reduce_sum(tf.cast(correct_preds,tf.float32))   # 数据类型转换并对预测值偏差求和
    y_score = []

    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds=0

    # 验证模型
    for i in range(n_batches):
        X_batch, Y_batch=mnist.test.next_batch(batch_size)
        preds_batch = sess.run([preds],feed_dict={X:X_batch,Y:Y_batch})
        y_score.append(preds_batch)
        # total_correct_preds += accuracy_batch[0]       

    # print ('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
    writer.close()

# 绘制ROC图像
with tf.Session() as sess:
    # Y = sess.run([],feed_dict={})
    fpr,tpr,threshold = roc_curve(y_score, mnist.test.labels)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()