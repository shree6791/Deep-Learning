import tensorflow as tf


class Network(object):

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):

        with tf.name_scope(layer_name):

            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])

            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])

            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                activations = act(preactivate, name='activation')

        return activations

    def feed_dict(self, dropout, keep_prob, mnist, train, x, y_):

        if train:
            xs, ys = mnist.train.next_batch(100)
            k = dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0

        return {x: xs, y_: ys, keep_prob: k}
