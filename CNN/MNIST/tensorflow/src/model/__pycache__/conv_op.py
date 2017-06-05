import os
import pandas as pd
import tensorflow as tf

max_steps = 10
num_epochs = None
batch_size = 100
min_queue_examples = 4
num_preprocess_threads = 1

test_folder = os.getcwd() + "\\test"
train_folder = os.getcwd() + "\\train"

# Initialize Data and Log Paths
log_dir = os.getcwd()  + '\\log'
#data_dir = os.getcwd()  + '/resource'

# Get Image Names
image_list = os.listdir(train_folder)

# Get Label Names
label_list = [x.split(".")[0] for x in image_list]
label_list = [ 1 if x=='cat' else 0 for x in label_list]

# Convert To Tensor
image = tf.convert_to_tensor(image_list)
label = tf.convert_to_tensor(label_list)

# Make an input queue

    
filename_queue = tf.train.slice_input_producer([image, label],
                    num_epochs=num_epochs, shuffle=True)

label = filename_queue[1]
file_contents = tf.read_file(filename_queue[0])
image = tf.image.decode_jpeg(file_contents, channels=3)

image.set_shape((28, 28, 3))

#label = tf.cast( label, tf.int64 )
#label = tf.one_hot( label, 2, 0, 1 )
#label = tf.cast( label, tf.float32 )

#return image, label

print(image)
print(label)

label.get_shape()

# Input Placeholder
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 3136], name = 'x-input')
    y = tf.placeholder(tf.float32, shape=[None, 2], name = 'y-output')
       
print("\nArchitcture Design:\n")

# Convolution Layer 1
with tf.name_scope('Convoltion_Layer_1'):   
    
    with tf.name_scope('weights_1'):
        
        W_conv1 = tf.truncated_normal([5, 5, 1, 32], stddev=0.1)
        W_conv1 = tf.Variable(W_conv1)
        
        b_conv1 = tf.constant(0.1, shape=[32])
        b_conv1 = tf.Variable(b_conv1)
            
    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1,28,28,1])
        
    # Convolution 1
    with tf.name_scope('Convoltion_1'):       
        h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
        h_conv1 = h_conv1 + b_conv1
                    
    # Activation
    with tf.name_scope('ReLu__1'):
        activation_1 = tf.nn.relu(h_conv1)
        
    print("Conv_1 Shape: " , activation_1.get_shape())
    
# Max Pooling Layer 1    
with tf.name_scope('Max_Pooling_1'):
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print("h_pool1 Shape: " , h_pool1.get_shape())
    
# Convolution Layer 2
with tf.name_scope('Convoltion_Layer_2'): 
    
    with tf.name_scope('weights_2'):
        W_conv2 = tf.truncated_normal([5, 5, 32, 64], stddev=0.1)
        W_conv2 = tf.Variable(W_conv2)
        
        b_conv2 = tf.constant(0.1, shape=[64])
        b_conv2 = tf.Variable(b_conv2)
    
    with tf.name_scope('Convoltion_Layer_2'):
        h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = h_conv2 + b_conv2       
        
    # Activation
    with tf.name_scope('ReLu__2'):
        activation_2 = tf.nn.relu(h_conv2)
        
    print("Conv_2 Shape: " , activation_2.get_shape())
    
# Sub Sampling    
with tf.name_scope('Max_Pooling_2'):    
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print("h_pool2 Shape: " , h_pool2.get_shape())
  
  
# Fully Connected Layer
with tf.name_scope('FC_1'):
    
    with tf.name_scope('weights_fc1'):
        W_fc1 = tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1)
        W_fc1 = tf.Variable(W_fc1)
        
        b_fc1 = tf.constant(0.1, shape=[1024])
        b_fc1 = tf.Variable(b_fc1)
    
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
        W_fc2 = tf.truncated_normal([1024, 2], stddev=0.1)
        W_fc2 = tf.Variable(W_fc2)
        
        b_fc2 = tf.constant(0.1, shape=[2])
        b_fc2 = tf.Variable(b_fc2)

    with tf.name_scope('weighted_sum'):
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
    print("FC_2 Shape: " , y_conv.get_shape())
    
# Image Summary
#W_conv1_summary = tf.summary.image("W1",W_conv1)
#W_conv2_summary = tf.summary.image("W2",W_conv2)
#W_fc1_summary = tf.summary.image("WFC1",W_fc1)
#W_fc2_summary = tf.summary.image("WFC2",W_fc2)

# Summary Log
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()

image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

# Launch Graph in Session

with tf.Session() as sess:
    
      
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(coord=coord)

    test_writer = tf.summary.FileWriter(log_dir + '/test')
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

    tf.global_variables_initializer().run()       
    
    print("\nTest Samples: ", image.get_shape())
    #print("Training Samples: ", len(mnist.train.images),"\n")
        
    for i in range(max_steps):

        
        print(i)
        ib, il = sess.run([image_batch, label_batch])
        
        #if i % 100 == 0:

            #[summary,test_accuracy] = sess.run([summary_op,accuracy],feed_dict={x:mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            #print("step %d, test accuracy %g"%(i, test_accuracy))
            #test_writer.add_summary(summary,i)     
            #test_writer.flush()                 


        if i % 100 == 0:

            [summary,train_accuracy] = sess.run([summary_op,accuracy],feed_dict={x:image_batch, y: label_batch, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            train_writer.add_summary(summary, i)
            train_writer.flush()

        train_op.run(feed_dict={x: ib, y: il, keep_prob: 0.9})
        
    #coord.request_stop()
    #coord.join(threads)

