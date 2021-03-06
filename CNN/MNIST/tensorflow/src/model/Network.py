import tensorflow as tf

class Network(object):
    
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
      
    def conv2d_Same(self,x, W):
        print(x)
        print(W)
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2_Same(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      
    def conv2d_Valid(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    
    def max_pool_2x2_Valid(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')