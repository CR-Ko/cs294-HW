
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt
from gym import wrappers

def main():

     def next_epoch(batch_size):
      return batch_size
     

     # Import data
     data = np.load('traindata.npz') 
     testdata = np.load('testdata.npz')
     print data.files 
     obs = data['observations'] 
     actions = data['actions']
     testobs = testdata['observations']
     testactions = testdata['actions']
     print('observations',obs.shape)
     print('actions',actions.shape)
     
     sess = tf.InteractiveSession()



     # Create the model
     x = tf.placeholder(tf.float32, shape=[None, 3]) #('observations', (1000, 11))
     W = tf.Variable(tf.zeros([3, 11]))
     b = tf.Variable(tf.zeros([11]))
     y = tf.matmul(x, W) + b
     #print y
     
     # Define loss and optimizer
     y_ = tf.placeholder(tf.float32, [None, 11])
     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
     train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
     sess = tf.InteractiveSession()
     tf.global_variables_initializer().run()
 
     start = 0
     end = 2000
     # Train
     for i in range(100):
       np.random.shuffle(actions)       
       #print(next_epoch(123))
       batch_xs = actions[start:end]  
       #print(batch_xs)
       #print('---')
       #batch_xs = batch_xs.transpose()        
       batch_xs = batch_xs.reshape(2000,3)
      # print(batch_xs)
       batch_ys = obs[start:end] 
       #batch_ys = batch_ys.reshape(11,1)
       #print(batch_xs.shape) 
       #print(batch_ys.shape)
       sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
     
     # Test trained model
     testobs = testobs[start:end]
     testactions = testactions[start:end]
     print('s',testactions.shape)
     testactions = testactions.reshape(1000,3)
     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     print(sess.run(accuracy, feed_dict={x: testactions,y_: testobs }))
     
 


if __name__ == '__main__':
    main()
