import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from model.Network import Network
from tensorflow.examples.tutorials.mnist import input_data

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)

# Reset Graph
tf.reset_default_graph()

# Initialize Constructor
nw = Network()

# Config Parameters
batch_size = 50
max_steps = 300


# Find Project Directory Path
#current_directory = os.path.dirname(__file__)
current_directory = os.path.dirname(os.path.realpath(__file__))
project_directory, _ = os.path.split(current_directory)

# Initialize Data and Log Paths
log_dir = project_directory + '/log'
data_dir = project_directory + '/resource'

# Delete Old Log Files
#if tf.gfile.Exists(log_dir):
#	tf.gfile.DeleteRecursively(log_dir)
#tf.gfile.MakeDirs(log_dir)

  
# Import Data
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# Input Placeholder
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784], name = 'x-input')
    y = tf.placeholder(tf.float32, shape=[None, 10], name = 'y-output')
       
print("\nArchitcture Design:\n")
    
# Convolution Layer 1
with tf.name_scope('Convoltion_Layer_1'):   
    
    with tf.name_scope('weights_1'):
        W_conv1 = nw.weight_variable([5, 5, 1, 32])
        b_conv1 = nw.bias_variable([32])
            
    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1,28,28,1])
        
    # Convolution 1
    with tf.name_scope('Convoltion_1'):
        h_conv1 = nw.conv2d_Same(x_image, W_conv1) + b_conv1        
        
    # Activation
    with tf.name_scope('ReLu__1'):
        activation_1 = tf.nn.relu(h_conv1)
        
    print("Conv_1 Shape: " , activation_1.get_shape())
    
    with tf.name_scope('Visualize_filters') as scope:
    
    # In this section, we visualize the filters of the first convolutional layers
    # We concatenate the filters into one image
    # Credits for the inspiration go to Martin Gorner
      W1_a = W_conv1                       # [5, 5, 1, 32]
      W1pad= tf.zeros([5, 5, 1, 1])        # [5, 5, 1, 4]  - four zero kernels for padding
      # We have a 6 by 6 grid of kernepl visualizations. yet we only have 32 filters
      # Therefore, we concatenate 4 empty filters
      W1_b = tf.concat(3, [W1_a, W1pad, W1pad, W1pad, W1pad])   # [5, 5, 1, 36]  
      W1_c = tf.split(3, 36, W1_b)         # 36 x [5, 5, 1, 1]
      W1_row0 = tf.concat(0, W1_c[0:6])    # [30, 5, 1, 1]
      W1_row1 = tf.concat(0, W1_c[6:12])   # [30, 5, 1, 1]
      W1_row2 = tf.concat(0, W1_c[12:18])  # [30, 5, 1, 1]
      W1_row3 = tf.concat(0, W1_c[18:24])  # [30, 5, 1, 1]
      W1_row4 = tf.concat(0, W1_c[24:30])  # [30, 5, 1, 1]
      W1_row5 = tf.concat(0, W1_c[30:36])  # [30, 5, 1, 1]
      W1_d = tf.concat(1, [W1_row0, W1_row1, W1_row2, W1_row3, W1_row4, W1_row5]) # [30, 30, 1, 1]
      W1_e = tf.reshape(W1_d, [1, 30, 30, 1])
      tf.summary.image("Visualize_kernels", W1_e)

# Max Pooling Layer 1    
with tf.name_scope('Max_Pooling_1'):
    h_pool1 = nw.max_pool_2x2_Same(h_conv1)
    print("h_pool1 Shape: " , h_pool1.get_shape())

# Convolution Layer 2
with tf.name_scope('Convoltion_Layer_2'): 
    
    with tf.name_scope('weights_2'):
        W_conv2 = nw.weight_variable([5, 5, 32, 64])
        b_conv2 = nw.bias_variable([64])
    
    with tf.name_scope('Convoltion_Layer_2'):
        h_conv2 = nw.conv2d_Same(h_pool1, W_conv2) + b_conv2       
        
    # Activation
    with tf.name_scope('ReLu__2'):
        activation_2 = tf.nn.relu(h_conv2)
        
    print("Conv_2 Shape: " , activation_2.get_shape())

# Sub Sampling    
with tf.name_scope('Max_Pooling_2'):    
    h_pool2 = nw.max_pool_2x2_Same(h_conv2)
    print("h_pool2 Shape: " , h_pool2.get_shape())
    
# Fully Connected Layer
with tf.name_scope('FC_1'):
    
    with tf.name_scope('weights_fc1'):
        W_fc1 = nw.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = nw.bias_variable([1024])
    
    with tf.name_scope('Un_Rolling'):    
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        
    print("h_pool2_flat Shape: " , h_pool2_flat.get_shape())
     
    with tf.name_scope('weighted_sum'):
        h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        
    with tf.name_scope('ReLu_3'):
        activation_3 = tf.nn.relu(h_fc1)
        
    print("FC_1 Shape: " , activation_3.get_shape())

# Drop Out
with tf.name_scope('Drop_Out'):
    keep_prob = tf.placeholder(tf.float32, name='dropout-probability')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Output Layer
with tf.name_scope('FC_2'):
    
    with tf.name_scope('weights_fc2'):
        W_fc2 = nw.weight_variable([1024, 10])
        b_fc2 = nw.bias_variable([10])

    with tf.name_scope('weighted_sum'):
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
    print("FC_2 Shape: " , y_conv.get_shape())

# Find Cross Entropy
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y))

# Train Model
with tf.name_scope('train'):    
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Find Accuracy
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Image Summary
#W_conv1_summary = tf.summary.image("W1",W_conv1)
#W_conv2_summary = tf.summary.image("W2",W_conv2)
#W_fc1_summary = tf.summary.image("WFC1",W_fc1)
#W_fc2_summary = tf.summary.image("WFC2",W_fc2)

# Summary Log
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()



# Launch Graph in Session

with tf.Session() as sess:

    
    tf.global_variables_initializer().run()

    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    
    print("\nTest Samples: ", len(mnist.test.images))
    print("Training Samples: ", len(mnist.train.images),"\n")
        
    for i in range(max_steps):
        
        batch = mnist.train.next_batch(batch_size)
        
        if i % 100 == 0:
            
            [summary,test_accuracy] = sess.run([summary_op,accuracy],feed_dict={x:mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            print("step %d, test accuracy %g"%(i, test_accuracy))
            test_writer.add_summary(summary,i)     
            test_writer.flush()                 
        
     
        if i % 100 == 0:
            
            [summary,train_accuracy] = sess.run([summary_op,accuracy],feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            train_writer.add_summary(summary, i)
            train_writer.flush()
            
        train_op.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.9})
        
        

    def getActivations(layer,image):
        units = sess.run(layer,feed_dict={x:np.reshape(image,[1,784],order='F'),keep_prob:1.0})
        plotNNFilter(units)
        
    
    def plotNNFilter(units):
        filters = units.shape[3]
        plt.figure(1, figsize=(10,10))
        n_columns = 8
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            plt.title('Filter ' + str(i))
            plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
            #plt.show() 
            
    print('\n\nTest Image:\n')
    imageToUse = mnist.test.images[0]
    #plt.imshow(np.reshape(imageToUse,[28,28]), interpolation="nearest", cmap="gray")
    #plt.show() 
    
    print('\nConvolution Layer 1 Filters:\n')
    #getActivations(activation_1,imageToUse)
    #plotNNFilter(W_conv1)
    #plt.show() 
    
    print('\nConvolution Layer 2 Filters:\n')
    #getActivations(activation_2,imageToUse)
    #plotNNFilter(W_conv2)
    #plt.show() 
    
    #print('Conv3 Filters Image:\n')
    #getActivations(activation_3,imageToUse)
    #plt.show() 
    
    #plt.tight_plot()
    print("End")
    
sess.close()