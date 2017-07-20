from __future__ import division

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

rounds = 100

class0=0
class1=9

layers = [7,5,1]
#import mnist data
mnist = input_data.read_data_sets('MNIST_data')

#create the model
x0 = tf.placeholder(tf.float32,[None,784])
x1 = tf.placeholder(tf.float32, [None,784])

#first layer######################################
W1 = tf.Variable(2*(np.random.rand(784,layers[0]))-1, dtype=np.float32)
#bias of the input layer
b1 = tf.Variable(tf.zeros([layers[0]]))
#output of the first layer
h1_0 = tf.matmul(x0,W1)+b1
h1_1 = tf.matmul(x1,W1)+b1

#second  layer#########################
W2 = tf.Variable(tf.ones([layers[0],layers[1]]))
b2 = tf.zeros([layers[1]])
h2_0 = tf.matmul(h1_0,W2)
# h2_0 = 1/(1+tf.exp(-h2_0)) + b2
h2_0 = tf.tanh(h2_0) + b2
h2_1 = tf.matmul(h1_1,W2)
# h2_1 = 1/(1+tf.exp(-h2_1)) + b2
h2_1 = tf.tanh(h2_1)+ b2


#third layer#########################
W3 = tf.Variable(tf.ones([layers[1],layers[2]]))
b3 = tf.zeros([layers[2]])
h3_0 = tf.matmul(h2_0,W3) + b3
# h3_0 = 1/(1+tf.exp(-h3_0)) + b3
h3_1 = tf.matmul(h2_1,W3) + b3
# h3_1 = 1/(1+tf.exp(-h3_1)) + b3


#compute MMD################################
mean_0 = tf.reduce_mean(h3_0)
mean_1 = tf.reduce_mean(h3_1)
mmd=mean_1-mean_0

#wanted = tf.exp(-mmd)
wanted = 10000-mmd
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(wanted)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

file_mmd = open('output/mmd.txt','w')
for j in range(rounds):
	x, y=mnist.train.next_batch(100)
	class0_y = []
	for i in range(len(y)):
		if y[i]==class0:
			class0_y.append(i)
	class1_y = []
	for i in range(len(y)):
		if y[i]==class1:
			class1_y.append(i)
	x_0 = x[class0_y]
	x_1 = x[class1_y]
	out = sess.run([train_step,mmd,h3_0,h3_1], feed_dict={x0: x_0, x1:x_1})
	if j%(rounds/10)==0:
		# file_mmd.write("class_0: %s class_1: %s\n" % (out[0], out[1]))
		file_mmd.write("mmd %s\n"%out[1])
		file0 = open('output/'+str(j)+'_score_0.txt','w')
		for score_0 in np.sort(out[2].ravel()):
			file0.write("%s\n"%score_0)
		file1 = open('output/'+str(j)+'_score_1.txt','w')
		for score_1 in np.sort(out[3].ravel()):
			file1.write("%s\n"%score_1)



#train all for test
x_train_for_test =mnist.train.images
y_train_for_test =mnist.train.labels
class0_y_test = []
class1_y_test = []
for i in range(len(y_train_for_test)):
	if y_train_for_test[i] == class0:
		class0_y.append(i)
	if y_train_for_test[i] == class1:
		class1_y.append(i)
x_0_for_test = x_train_for_test[class0_y_test]
x_1_for_test = x_train_for_test[class1_y_test]
test_metric = sess.run([mean_0, mean_1], feed_dict = {x0:x_0, x1:x_1})

# print("trained all")
# print(test_metric[0], test_metric[1])
#test################################################
correct = tf.reduce_sum(tf.cast(tf.less(h3_0,0.0),tf.float32)) + tf.reduce_sum(tf.cast(tf.less(0.0,h3_1),tf.float32))
num = tf.size(h3_1)+tf.size(h3_0)
tf.cast(num, tf.float32)
index_0 = np.equal(class0,mnist.test.labels)
x_00 = mnist.test.images[index_0]
index_1 = np.equal(class1,mnist.test.labels)
x_11 = mnist.test.images[index_1]
out = sess.run([correct,num, h3_0, h3_1, x0, x1], feed_dict={x0: x_00, x1:x_11})

print("test resultss")
print(out[0],out[1])
print("accuracy")
print(out[0]/out[1])

test0_score = open('output/test0.txt','w')
for score_test0 in out[2]:
	test0_score.write("%s\n"%score_test0)
test1_score = open('output/test1.txt','w')
for score_test1 in out[3]:
	test1_score.write("%s\n"%score_test1)
test_item = open('output/test_item00.txt','wb')
np.savetxt(test_item, out[4][0], delimiter="\r\n")