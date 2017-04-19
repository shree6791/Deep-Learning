
import string
import operator
import numpy as np

import os
import sys
sys.path.append(os.path.abspath('..'))

def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)

def remove_punctuation(s):
    return s.translate(string.punctuation)

def my_tokenizer(s):
    
    s = remove_punctuation(s)
    s = s.lower() # downcase
    return s.split()

def get_robert_frost_data(n_vocab):

    # return variables
    sentences = []
    current_idx = 2
    idx2word = ['START', 'END']
    word2idx = {'START': 0, 'END': 1}
    word_idx_count = {0: float('inf'), 1: float('inf')}

    for line in open('../resource/robert_frost.txt'):
		
        line = line.strip()
        
        if line:
		
            sentence = []
            tokens = my_tokenizer(line)
    			
            for t in tokens:
                
                if t not in word2idx:
                    
                    word2idx[t] = current_idx
                    idx2word.append(t)
                    current_idx += 1
    					
                idx = word2idx[t]
                word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
    				
            sentence_by_idx = [word2idx[t] for t in tokens]
            sentences.append(sentence_by_idx)

    # restrict vocab size
    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    
    new_idx = 0
    word2idx_small = {}    
    idx_new_idx_map = {}
	
    for idx, count in sorted_word_idx_count[:n_vocab]:
        
        word = idx2word[idx]
        word2idx_small[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        
        new_idx += 1
		
    # let 'unknown' be the last token
    word2idx_small['UNKNOWN'] = new_idx 
    unknown = new_idx

    # map old idx to new idx
    sentences_small = []
	
    for sentence in sentences:
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)

    return sentences_small, word2idx_small

