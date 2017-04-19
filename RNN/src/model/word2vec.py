
import json
import theano
import numpy as np
import theano.tensor as T
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.utils import shuffle
#from sklearn.decomposition import PCA


import os
import sys
sys.path.append(os.path.abspath('..'))
from helper.util import get_robert_frost_data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def init_weights(shape):
    return np.random.randn(*shape).astype(np.float32) / np.sqrt(sum(shape))


class Model(object):
    
    def __init__(self, D, V, context_sz):
        
        self.V = V # vocab size    
        self.D = D # embedding dimension
        self.Pnw = np.zeros(self.V)
        self.context_sz = context_sz # 2*context_sz

    def _get_pnw(self, X):
       
        word_freq = {}
        word_count = sum(len(x) for x in X)
        
        for x in X:
            for xj in x:
                if xj not in word_freq:
                    word_freq[xj] = 0
                word_freq[xj] = word_freq[xj] + 1
                
                
        for j in range(2, self.V):
            self.Pnw[j] = (word_freq[j] / float(word_count))**0.75
            
        return self.Pnw

    def _get_negative_samples(self, context, num_neg_samples):
        
        # temporarily save context values because we don't want to negative sample these
        saved = {}
        
        for context_idx in context:
            saved[context_idx] = self.Pnw[context_idx]            
            self.Pnw[context_idx] = 0
            
        neg_samples = np.random.choice(
            range(self.V),
            size=num_neg_samples, # this is arbitrary - number of negative samples to take
            replace=False,
            p=self.Pnw / np.sum(self.Pnw),
        )
        
        for j, pnwj in saved.items():
            self.Pnw[j] = pnwj          
        
        return neg_samples

    
    def fit(self, X, num_neg_samples=10, learning_rate=10e-5, mu=0.99, reg=0.1, epochs=10):
        
        N = len(X)
        V = self.V
        D = self.D

        # Get -ve sample distribution
        self._get_pnw(X) 

        # initialize weights and momentum changes
        W1 = init_weights((V, D))
        W2 = init_weights((D, V))
        W1 = theano.shared(W1)
        W2 = theano.shared(W2)

        thInput = T.iscalar('input_word')
        thContext = T.ivector('context')
        thNegSamples = T.ivector('negative_samples')

        W1_subset = W1[thInput]
        W2_psubset = W2[:, thContext]
        W2_nsubset = W2[:, thNegSamples]
        
        p_activation = W1_subset.dot(W2_psubset)
        pos_pY = T.nnet.sigmoid(p_activation)
        
        n_activation = W1_subset.dot(W2_nsubset)
        neg_pY = T.nnet.sigmoid(-n_activation)
        
        cost = -T.log(pos_pY).sum() - T.log(neg_pY).sum()

        W1_grad = T.grad(cost, W1_subset)
        W2_pgrad = T.grad(cost, W2_psubset)
        W2_ngrad = T.grad(cost, W2_nsubset)

        W1_update = T.inc_subtensor(W1_subset, -learning_rate*W1_grad)
        W2_update = T.inc_subtensor(
            T.inc_subtensor(W2_psubset, -learning_rate*W2_pgrad)[:,thNegSamples], -learning_rate*W2_ngrad)
        
        updates = [(W1, W1_update), (W2, W2_update)]

        train_op = theano.function(
            inputs=[thInput, thContext, thNegSamples],
            outputs=cost,
            updates=updates,
            allow_input_downcast=True,
        )

        costs = []
        cost_per_epoch = []
        sample_indices = range(N)
        
        for i in range(epochs):
            
            t0 = datetime.now()
            cost_per_epoch_i = []
            sample_indices = shuffle(sample_indices)
                        
            for it in range(N):
                
                j = sample_indices[it]
                x = X[j] # one sentence

                # too short to do 1 iteration, skip
                if len(x) < 2 * self.context_sz + 1:
                    continue

                cj = []
                n = len(x)
                
                for jj in range(n):

                     start = max(0, jj - self.context_sz)
                     end = min(n, jj + 1 + self.context_sz)
                     
                     context = np.concatenate([x[start:jj], x[(jj+1):end]])
                     context = np.array(list(set(context)), dtype=np.int32)
                     
                     neg_samples = self._get_negative_samples(context, num_neg_samples)

                     c = train_op(x[jj], context, neg_samples)
                     cj.append(c / (num_neg_samples + len(context)))

               

                cj = np.mean(cj)
                cost_per_epoch_i.append(cj)
                costs.append(cj)


            epoch_cost = np.mean(cost_per_epoch_i)
            cost_per_epoch.append(epoch_cost)
            print ("Iteration %d:" % i, " Cost: ", epoch_cost)

        self.W1 = W1.get_value()
        self.W2 = W2.get_value()

        plt.figure()
        plt.plot(cost_per_epoch)
        plt.title("Cost Function (per epoch)")
        plt.show()
        
def visualizeEmbeddings(we_file, w2i_file, Model):
    
    plt.figure()
    plt.show()
    plt.title("Word Embeddings")
    
    We = np.load(we_file)
    V, D = We.shape
    
    with open(w2i_file) as f:
        word2idx = json.load(f)

    idx2word = {v:k for k,v in word2idx.items()}

    model = Model()
    Z = model.fit_transform(We)
    plt.scatter(Z[:,0], Z[:,1])
    
    for i in range(V):
        plt.annotate(s=idx2word[i], xy=(Z[i,0], Z[i,1]))
                    
    
def main(we_file,w2i_file):
    
    sentences, word2idx = get_robert_frost_data(n_vocab=1200)
	
    with open(w2i_file, 'w') as f:
        json.dump(word2idx, f)

    V = len(word2idx)    
    model = Model(50, V, 5)
    model.fit(sentences, learning_rate=10e-4, mu=0, epochs=10000, num_neg_samples=5)
    
    W1 = model.W1
    W2 = model.W2
    
    We = (W1 + W2.T) / 2
    np.save(we_file, We)
        

if __name__ == '__main__':
    
    file_path = os.path.abspath('..')+ "\\resource\\"
    we_file = file_path + 'w2v_word_embedding.npy'       # word embedding file
    w2i_file = file_path + 'w2v_word2idx.json'           # word 2 index file
    
    
    main(we_file, w2i_file)
    
    visualizeEmbeddings(we_file, w2i_file, Model=TSNE)
 
   