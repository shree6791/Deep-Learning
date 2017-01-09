import os
import tensorflow as tf
from model.Network import Network
from model.Helper import Helper
from tensorflow.examples.tutorials.mnist import input_data

# Initialize Helper Class
h = Helper()
nw = Network()

# Initialize Variables
dropout = 0.9
max_steps = 1000
learning_rate = 0.001

# Find Project Directory Path
current_directory = os.path.dirname(__file__)
project_directory, _ = os.path.split(current_directory)

# Initialize Data and Log Paths
log_dir = project_directory + '/log'
data_dir = project_directory + '/resource'

# Delete Old Log Files
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)

# Import data
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# Input Place Holders
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

# Display Image
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

# 1st Hidden Layer
hidden1 = nw.nn_layer(x, 784, 500, 'layer1')

# Dropout
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    dropped = tf.nn.dropout(hidden1, keep_prob)

# Output Layer
y_ = nw.nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

# Find Cross Entropy
with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

# Train Model
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        cross_entropy)

# Accuracy of Model
with tf.name_scope('accuracy'):

    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Log the Output
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)

# Merge All Summaries
summary_op = tf.summary.merge_all()

# Launch Graph In Session
with tf.Session() as sess:

    tf.global_variables_initializer().run()

    # Initialize Log Paths
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    for i in range(max_steps):

        # Print Test Accuracy and Log Test Parameters
        if i % 50 == 0:

            summary, acc = sess.run(
                [summary_op, accuracy], feed_dict=nw.feed_dict(dropout, keep_prob, mnist, False, x, y))

            print('Test Accuracy @ Step %s: %s' % (i, acc))
            test_writer.add_summary(summary, i)

        # Log Train Parameters
        summary, _ = sess.run(
            [summary_op, train_step], feed_dict=nw.feed_dict(dropout, keep_prob, mnist, True, x, y))
        train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()
