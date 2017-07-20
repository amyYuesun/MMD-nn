from __future__ import division

import tensorflow as tf
import numpy as np

#number of steps (should be a multiple of 10 because we write training results every rounds/10 steps)
rounds = 10
#size of batches from the original training set (including examples from other classes)
batch_size = 20
#feature of the examples
feature = 4
#two classes labels
class0 = 2
class1 = 0

#model structure
layers = [7,5,1]
#import iris data
iris_train = tf.contrib.learn.datasets.base.load_csv_with_header(
	filename="iris_training.csv",target_dtype=np.int,features_dtype=np.float32)

#create the model
x0 = tf.placeholder(tf.float32,[None,feature])
x1 = tf.placeholder(tf.float32, [None,feature])

#first layer######################################
#weights of input layer
W1 = tf.Variable(2*(np.random.rand(feature,layers[0]))-1, dtype=np.float32)
b1 = tf.Variable(tf.zeros([layers[0]]))

h1_0 = tf.matmul(x0,W1)+b1
h1_1 = tf.matmul(x1,W1)+b1

#second  layer#########################
W2 = tf.Variable(tf.ones([layers[0],layers[1]]))
b2 = tf.zeros([layers[1]])

h2_0 = tf.matmul(h1_0,W2)
h2_1 = tf.matmul(h1_1,W2)


#third layer#########################
W3 = tf.Variable(tf.ones([layers[1],layers[2]]))
b3 = tf.zeros([layers[2]])

h3_0 = tf.matmul(h2_0,W3) + b3
h3_1 = tf.matmul(h2_1,W3) + b3


#compute MMD################################
mean_0 = tf.reduce_mean(h3_0)
mean_1 = tf.reduce_mean(h3_1)
mmd=mean_1-mean_0

#training step to minimize(10000-mmd), that is, maximize mmd##################################################
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(10000-mmd)


#training process###########################################################################################
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_size = len(iris_train.data)
file_mmd = open('output/mmd.txt','w')

for j in range(rounds):
	#randomly choose examples for each step
	chosen_index = np.random.choice(train_size,batch_size)
	x= iris_train.data[chosen_index]
	y= iris_train.target[chosen_index]
	#only keep the examples that belong to the two classes that we are interested in
	class0_y = []
	class1_y = []
	for i in range(len(y)):
		if y[i]==class0:
			class0_y.append(i)
		if y[i]==class1:
			class1_y.append(i)
	x_0 = x[class0_y]
	x_1 = x[class1_y]
	#run the training steps
	out = sess.run([train_step,mmd,h3_0,h3_1], feed_dict={x0: x_0, x1:x_1})
	#print out the outputs; the "score" given by the model for each example at jth step
	if (j+1)%(rounds/10)==0:
		file_mmd.write("mmd %s\n"%out[1])
		file0 = open('output/'+str(j)+'_score_0.txt','w')
		for score_0 in np.sort(out[2].ravel()):
			file0.write("%s\n"%score_0)
		file1 = open('output/'+str(j)+'_score_1.txt','w')
		for score_1 in np.sort(out[3].ravel()):
			file1.write("%s\n"%score_1)

#compute the final model on training data set for final scores################################################################
x_train_for_test = iris_train.data
y_train_for_test = iris_train.target
class0_y_test = []
class1_y_test = []
for i in range(len(y_train_for_test)):
	if y_train_for_test[i]==class0:
		class0_y_test.append(i)
	if y_train_for_test[i]==class1:
		class1_y_test.append(i)
x_0_for_test = x_train_for_test[class0_y_test]
x_1_for_test = x_train_for_test[class1_y_test]
#mean_0 is the mean score given by the final model on all examples belong to class0
test_metric = sess.run([mean_0, mean_1], feed_dict={x0:x_0_for_test, x1:x_1_for_test})
# print("metrics after all")
# print(test_metric[0], test_metric[1])


#test the model on the test set#########################################################################
mean0 = tf.constant(test_metric[0])
mean1 = tf.constant(test_metric[1])
#compare the distance to the mean of class0 and class1 to predict the test item
dis_0_0 = tf.abs(h3_0-mean0)
dis_0_1 = tf.abs(h3_0-mean1)
correct0 = tf.reduce_sum(tf.cast(tf.less(dis_0_0, dis_0_1), dtype=tf.float32))
dis_1_0 = tf.abs(h3_1-mean0)
dis_1_1 = tf.abs(h3_1-mean1)
correct1 = tf.reduce_sum(tf.cast(tf.less(dis_1_1, dis_1_0), dtype=tf.float32))
correct = correct0+correct1


num = tf.size(h3_1)+tf.size(h3_0)
tf.cast(num, tf.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
	filename="iris_test.csv",target_dtype=np.int,features_dtype=np.float32)

#select the examples that belong to the two classes that we are interested in
index_0 = np.equal(class0,test_set.target)
x_00 = test_set.data[index_0]
index_1 = np.equal(class1,test_set.target)
x_11 = test_set.data[index_1]
#feed the test set into the model
out = sess.run([correct,num, h3_0, h3_1, mean0, mean1, x0], feed_dict={x0: x_00, x1:x_11})

#print out test results
print("test results")
print(out[0],out[1])
# print(out[4], out[5])
test0_score = open('output/test0.txt','w')
for score_test0 in out[2]:
	test0_score.write("%s\n"%score_test0)
test1_score = open('output/test1.txt','w')
for score_test1 in out[3]:
	test1_score.write("%s\n"%score_test1)
