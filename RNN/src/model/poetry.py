
import theano
import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from helper.util import init_weight, get_robert_frost_data

import os
import sys
sys.path.append(os.path.abspath('..'))


class SimpleRNN:

    def __init__(self, D, M, V):
        
        self.D = D # dimensionality of word embedding
        self.M = M # hidden layer size
        self.V = V # vocabulary size

		
    def fit(self, We, X, learning_rate=10e-1, mu=0.99, reg=1.0, activation=T.tanh, epochs=500, show_fig=False):
	
        N = len(X)
        D = self.D
        M = self.M
        V = self.V
        self.f = activation
        
        
        # initial weights
        We = np.float64(We)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, V)
        bo = np.zeros(V)

        # make them theano shared
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thY = T.ivector('Y')
        thX = T.ivector('X')
        Ei = self.We[thX] # will be a TxD matrix
      

        def recurrence(x_t, h_t1):
            
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=Ei,
            n_steps=Ei.shape[0],
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

#        print(updates)
        
        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates
        )

        costs = []
        n_total = sum((len(sentence)+1) for sentence in X)
		
        for i in range(epochs):
            
            X = shuffle(X)
            n_correct = 0
            cost = 0
            
            for j in range(N):
                
                # we set 0 to start and 1 to end
                input_sequence = [0] + X[j]
                output_sequence = X[j] + [1]
               
                
                c, p = self.train_op(input_sequence, output_sequence)
                
                cost += c
                
                for pj, xj in zip(p, output_sequence):
                    
                    if pj == xj:
                        n_correct += 1
                        
            print ("i:", i, "cost:", cost, "correct rate:", (float(n_correct)/n_total))
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()  
  
  
    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])

		
    @staticmethod
    def load(filename, activation):
	
        # TODO: would prefer to save activation to file too
        npz = np.load(filename)
        
        We = npz['arr_0']
        Wx = npz['arr_1']
        Wh = npz['arr_2']
        bh = npz['arr_3']
        h0 = npz['arr_4']
        Wo = npz['arr_5']
        bo = npz['arr_6']
		
        _, M = Wx.shape
        V, D = We.shape
                
        rnn = SimpleRNN(D, M, V)
        rnn.set(We, Wx, Wh, bh, h0, Wo, bo, activation)
		
        return rnn

	
	
    def set(self, We, Wx, Wh, bh, h0, Wo, bo, activation):
	
        self.f = activation

        # redundant - see how you can improve it
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thY = T.ivector('Y')
        thX = T.ivector('X')
        Ei = self.We[thX] # will be a TxD matrix
        

        def recurrence(x_t, h_t1):
            
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=Ei,
            n_steps=Ei.shape[0],
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=prediction,
            allow_input_downcast=True,
        )

	
	
    def generate(self, pi, word2idx):
	
        # convert word2idx -> idx2word
        idx2word = {v:k for k,v in word2idx.items()}
        V = len(pi)

        n_lines = 0

        # why? because using the START symbol will always yield the same first word!
        X = [ np.random.choice(V, p=pi) ]
        print (idx2word[X[0]], end=" ")

        while n_lines < 4:
            
            P = self.predict_op(X)[-1]
            X += [P]
            
            if P > 1:
                word = idx2word[P]
                print(word, end=" ")
                
            elif P == 1:
                
                n_lines += 1
                print('')
                
                if n_lines < 4:
                    X = [ np.random.choice(V, p=pi) ] # reset to start of line
                    print (idx2word[X[0]], end=" ")

        
					
def train_poetry(rnn_file, we_file):    
     
    We = np.load(we_file)
    V, D = We.shape

    sentences, word2idx = get_robert_frost_data(n_vocab=1200)
    
    rnn = SimpleRNN(D, 30, V)	
    rnn.fit(We, sentences, learning_rate=10e-5, show_fig=True, activation=T.nnet.relu, epochs=2000)
    
    rnn.save(rnn_file)

	
def generate_poetry(rnn_file):
    
    sentences, word2idx = get_robert_frost_data(n_vocab=1200)
    rnn = SimpleRNN.load(rnn_file, T.nnet.relu)
    
    V = len(word2idx)
    pi = np.zeros(V)
	
    for sentence in sentences:
        pi[sentence[0]] += 1
		
    pi /= pi.sum()

    rnn.generate(pi, word2idx)


if __name__ == '__main__':
    
    file_path = os.path.abspath('..')+ "\\resource\\"
    
    w2i_file = file_path + 'w2v_word2idx.json'           # word 2 index file
    we_file = file_path + 'w2v_word_embedding.npy'       # word embedding file
    
    rnn_file = file_path + 'robert_frost_rnn.npz'
    
    #train_poetry(rnn_file, we_file)
    generate_poetry(rnn_file)    
	
	