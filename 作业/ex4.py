import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier 


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

    for i in range(n_epochs):

        total_loss=0

        for _ in range(n_batches):
            X_batch, Y_batch =mnist.train.next_batch(batch_size)
            _,loss_batch =sess.run([optimizer,loss],feed_dict={X:X_batch,Y:Y_batch})
            total_loss +=loss_batch
        print ('Average loss epoch {0}:{1}'.format(i,total_loss/n_batches))
    print ('Total time: {0} seconds'.format(time.time()-start_time))

    print ('optimizatin Finished')


    preds = tf.nn.softmax(logits)
    correct_preds=tf.equal(tf.argmax(preds,1),tf.argmax(Y,1))
    accuracy=tf.reduce_sum(tf.cast(correct_preds,tf.float32))

    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds=0

    for i in range(n_batches):
        X_batch, Y_batch=mnist.test.next_batch(batch_size)
        accuracy_batch =sess.run([accuracy],feed_dict={X:X_batch,Y:Y_batch})
        total_correct_preds += accuracy_batch[0]

    print ('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))

    writer.close()