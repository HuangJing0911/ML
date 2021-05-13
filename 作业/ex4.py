import imp
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_array_ops import reshape
from scipy import interp
from itertools import cycle

# ROC绘图
def plot_roc(Y_valid, Y_pred, nb_classes, filepath):
    # roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
    # 横坐标：假正率（False Positive Rate , FPR）
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= nb_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
    label='micro-average ROC curve (area = {0:0.2f})'
    ''.format(roc_auc["micro"]),
    color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
    label='macro-average ROC curve (area = {0:0.2f})'
    ''.format(roc_auc["macro"]),
    color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
        label='ROC curve of class {0} (area = {1:0.2f})'
        ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(filepath)
    plt.show()

# ex4_1:多项逻辑斯蒂回归的编程实现
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
batch_size=128
X = tf.placeholder(tf.float32,[None,784],name='X_placeholder')
Y = tf.placeholder(tf.int32, [None,10],name='Y_placehoder')

# ex4_1_1:普通梯度下降logistic回归
def normal_logistic():

    w = tf.Variable(tf.random_normal(shape=[784,10],stddev=0.01),name='weights')
    b = tf.Variable(tf.zeros([1,10]),name='bias')

    # W*x+b
    logits=tf.matmul(X,w)+b
    tf.nn.weighted_cross_entropy_with_logits
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
        correct_preds=tf.equal(tf.argmax(preds,1),tf.argmax(Y,1))   # 找到预测值preds和Y的每一行最大的索引并逐个判断
        accuracy=tf.reduce_sum(tf.cast(correct_preds,tf.float32))   # 数据类型转换并对预测值偏差求和
        y_score = [] # list

        n_batches = int(mnist.test.num_examples/batch_size)
        total_correct_preds=0

        # 验证模型
        for i in range(n_batches):
            X_batch, Y_batch=mnist.test.next_batch(batch_size)
            preds_batch = sess.run([preds],feed_dict={X:X_batch,Y:Y_batch})
            # print(i,np.array(preds_batch[0]).shape)
            y_score.append(np.array(preds_batch[0]).tolist())
            # total_correct_preds += accuracy_batch[0]       

        # print ('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
        writer.close()
        # print(y_score)  # <class 'numpy.ndarray'>
        y_score = np.array(y_score)
        y_score = y_score.reshape((n_batches*batch_size, 10))
        print("len of y_score is:(",len(y_score),len(y_score[0]),")")

    # 绘制ROC图像 
    with tf.Session() as sess:
        # 对测试数据结果进行转换
        Y_pred = tf.arg_max(y_score,1)                          # 取出预测值中元素最大值所对应的索引
        Y_valid = tf.arg_max(mnist.test.labels, 1)              # 取出Y中元素最大值对应的索引
        Y1, Y2 = sess.run([Y_pred,Y_valid])
        Y_pred = np.array(Y1)
        Y2 = Y2[0:n_batches*batch_size]
        Y_valid = np.array(Y2)
        print(Y1.shape,Y2.shape)
        # 对数据进行二值化
        class_of_mnist = [i for i in range(10)]
        Y_pred = label_binarize(Y_pred, classes = class_of_mnist)
        Y_valid = label_binarize(Y_valid, classes = class_of_mnist)
        plot_roc(Y_valid, Y_pred, 10, "./images/ROC/ROC_10分类.png")    # 具体绘制图片

# ex4_1_2:使用l1正则化的logistic回归
def regularize_logistic():

    w = tf.Variable(tf.random_normal(shape=[784,10],stddev=0.01),name='weights')
    b = tf.Variable(tf.zeros([1,10]),name='bias')