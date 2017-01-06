# Variables Used
#    X = Input Data set
#    Y = Actual Output
#    Y_ = Predicted Output


import tensorflow as tf


class Helper():

    def combine_inputs(self, X, W, b):
        return tf.matmul(X, W) + b

    def inference(self, X, W, b):
        return tf.nn.softmax(self, self.combine_inputs(X, W, b))

    def cross_entropy(self, X, W, b, Y):
        Y_ = self.combine_inputs(X, W, b)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(Y_, Y))

    def train(self, learning_rate, total_loss):
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

    def evaluate(self, X, Y):
        Y_ = tf.cast(tf.arg_max(self.inference(X), 1), tf.int32)
        return tf.reduce_mean(tf.cast(tf.equal(Y_, Y), tf.float32))
