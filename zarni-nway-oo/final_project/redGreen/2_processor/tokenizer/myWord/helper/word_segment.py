"""

This code is updated version of this: https://gist.github.com/markdtw/e2a4e2ee7cef8ea6aed33bb47a97fba6
Ye Kyaw Thu, LST, NECTEC, Thailand updated followings:
-- added recursion limit
-- changed P_unigram and P_bigram as module level global variable
-- using binary ngram dictionary
--  set N value of this: "def __init__(self, datafile=None, unigram=True, N=102490):"
-- Last Updated: 5 Sept 2021

# References:
- Python implementation of Viterbi algorithm for word segmentation: 
- Updated version of this: https://gist.github.com/markdtw/e2a4e2ee7cef8ea6aed33bb47a97fba6
- A clean-up of this: http://norvig.com/ngrams/ch14.pdf
- For recursion limit: https://www.geeksforgeeks.org/python-handling-recursion-limit/
- A. Viterbi, "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm," in IEEE Transactions on Information Theory, vol. 13, no. 2, pp. 260-269, April 1967, doi: 10.1109/TIT.1967.1054010.

"""

import math
import functools
import sys
import pickle


sys.setrecursionlimit(10**6)

global P_unigram
global P_bigram

def read_dict (fileDICT):
    try:
        with open(fileDICT, 'rb') as input_file:
            dictionary = pickle.load(input_file)
            input_file.close()
    except FileNotFoundError:
        print('Dictionary file', fileDICT, ' not found!')
    return dictionary

class ProbDist(dict):
    ### Probability distribution estimated from unigram/bigram data
    def __init__(self, datafile=None, unigram=True, N=102490):
    #def __init__(self, datafile=None, unigram=True, N=1024908267229):
    #def __init__(self, datafile=None, unigram=True, N=8199266137832):
        #data = {}
        data = read_dict(datafile)
        for k, c in data.items():
            self[k] = self.get(k, 0) + c

        if unigram:
            self.unknownprob = lambda k, N: 10/(N*10**len(k))    # avoid unknown long word
        else:
            self.unknownprob = lambda k, N: 1/N

        self.N = N

    def __call__(self, key):
        if key in self:
            return self[key]/self.N
        else:
            return self.unknownprob(key, self.N)


def conditionalProb(word_curr, word_prev):
    ### Conditional probability of current word given the previous word.
    try:
        return P_bigram[word_prev + ' ' + word_curr]/P_unigram[word_prev]
    except KeyError:
        return P_unigram(word_curr)


@functools.lru_cache(maxsize=2**16)  # Increased cache from 1024 to 65536 entries
#maxlen=20
def viterbi(text, prev='<S>', maxlen=20):
    if not text:
        return 0.0, []
    
    # Optimize for short texts - no need for complex processing
    if len(text) <= 3:
        return math.log10(conditionalProb(text, prev)), [text]
    
    textlen = min(len(text), maxlen)
    splits = [(text[:i + 1], text[i + 1:]) for i in range(textlen)]

    candidates = []
    best_score = float('-inf')
    best_candidate = None
    
    # Early stopping optimization - keep track of best score
    for first_word, remain_word in splits:
        first_prob = math.log10(conditionalProb(first_word, prev))
        
        # Skip if this first word probability is too low (pruning)
        if first_prob < -10:  # Threshold for pruning unlikely candidates
            continue
            
        remain_prob, remain_words = viterbi(remain_word, first_word)
        total_prob = first_prob + remain_prob
        candidate = (total_prob, [first_word] + remain_words)
        
        # Keep track of best candidate for early stopping
        if total_prob > best_score:
            best_score = total_prob
            best_candidate = candidate
        
        candidates.append(candidate)
        
        # Early stopping if we have a very good candidate
        if total_prob > -2:  # Good enough threshold
            break
    
    return best_candidate if best_candidate else max(candidates) if candidates else (0.0, [text])
