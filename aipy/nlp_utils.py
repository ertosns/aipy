import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

import pickle

##Process tweet function.
#
# @param tweet a string containing a tweet
# @return tweets_clean a list of words containing the processed tweet
def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

## Build frequencies.
#
# @param (m) tweets a list of tweets
# @param ys an m x 1 array with the sentiment label of each tweet             (either 0 or 1)
# @return freqs a dictionary mapping each (word, sentiment) pair to its frequency
def build_freqs(tweets, ys):
    
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()
    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
                
    return freqs

##
#
# @param freqs a dictionary with the frequency of each pair (or tuple)
# @param word the word to look up
# @param label the label corresponding to the word
# @return n the number of times the word with its corresponding label appears.
def lookup(freqs, word, label):    
    n = 0  # freqs.get((word, label), 0)

    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]

    return n


##
# @param freqs dictionary from (word, label) to how often the word appears
# @param train_x a list of tweets
# @param train_y a list of labels correponding to the tweets (0,1)
# @return logprior the log prior.
# @return loglikelihood the log likelihood of you Naive bayes.
def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0
    
    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    
    # calculate N_pos and N_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[(pair)]
        else:
            N_neg += freqs[(pair)]

    D = len(train_y)
    D_pos = sum(train_y)
    D_neg = D-D_pos

    # Calculate logprior
    logprior = np.log(D_pos/float(D_neg))
    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = freqs.get((word,1), 0)
        freq_neg = freqs.get((word,0), 0)
        # calculate the probability that each word is positive, and negative
        p_w_pos = float(freq_pos+1)/(N_pos+V)
        p_w_neg = float(freq_neg+1)/(N_neg+V)
        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)

    return logprior, loglikelihood



##
# @param tweet a string
# @param logprior a number
# @param loglikelihood a dictionary of words mapping test_o numbers
# @return p the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)
def naive_bayes_predict(tweet, logprior, loglikelihood):
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)
    # initialize probability to zero
    p = 0
    # add the logprior
    p += logprior

    for word in word_l:
        
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood.get(word, 0)
            
    return p

##
# @param freqs dictionary containing the words
# @param word string to lookup
# @param Output a dictionary with keys 'positive', 'negative', and 'ratio'.
#        Example: {'positive': 10, 'negative': 20, 'ratio': 0.5}
#
def get_ratio(freqs, word):
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    pos_neg_ratio['positive'] = lookup(freqs, word, 1)
    pos_neg_ratio['negative'] = lookup(freqs, word, 0)
  
    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive']+1)/(pos_neg_ratio['negative']+1)
    return pos_neg_ratio

##
# @param freqs dictionary of words
# @param label 1 for positive, 0 for negative
# @param threshold ratio that will be used as the cutoff for including a word in the returned dictionary
#
# @param word_set dictionary containing the word and information on its positive count, negative count, and ratio of positive to negative counts.
#        example of a key value pair:
#        {'happi':
#            {'positive': 10, 'negative': 20, 'ratio': 0.5}
#        }
def get_words_by_threshold(freqs, label, threshold):
    word_list = {}
    for key in freqs.keys():
        word, _ = key
        
        # get the positive/negative ratio for a word
        pos_neg_ratio = get_ratio(freqs, word)

        # if the label is 1 and the ratio is greater than or equal to the threshold...
        if label == 1 and pos_neg_ratio['ratio']>=threshold:

            # Add the pos_neg_ratio to the dictionary
            word_list[word]=pos_neg_ratio

        # If the label is 0 and the pos_neg_ratio is less than or equal to the threshold...
        elif label == 0 and pos_neg_ratio['ratio']<=threshold:

            # Add the pos_neg_ratio to the dictionary
            word_list[word]=pos_neg_ratio

    return word_list

## Calculate cosine similarities
# @param A a numpy array which corresponds to a word vector
# @param B A numpy array which corresponds to a word vector
# @return cos numerical number representing the cosine similarity between A and B.
def cosine_similarity(A, B):
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B) 
    cos = dot/(norma*normb)
    
    return cos


## Calculate Euclidean distance
# @param A a numpy array which corresponds to a word vector
# @param B A numpy array which corresponds to a word vector
# @return d numerical number representing the Euclidean distance between A and B.
def euclidean(A, B):
    d = np.linalg.norm(A-B)
    return d


## Compute principal component analysis.
#
# @param X of dimension (m,n) where each row corresponds to a word vector
# @param n_components Number of components you want to keep.
#
# @return X_reduced data transformed in 2 dims/columns + regenerated original data
def compute_pca(X, n_components=2):
    X_demeaned = X - np.mean(X, axis=0)
    covariance_matrix = np.cov(X_demeaned, rowvar=False)
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)

    idx_sorted = np.argsort(eigen_vals)    
    idx_sorted_decreasing = idx_sorted[::-1]

    eigen_vecs_sorted = eigen_vecs[:,idx_sorted_decreasing]
    eigen_vecs_subset = eigen_vecs_sorted[:,0:n_components]

    X_reduced = np.dot(eigen_vecs_subset.T, X_demeaned.T).T

    return X_reduced


## Compute Loss function
# using least square method.
# @param X a matrix of dimension (m,n) where the columns are the English embeddings.
# @param Y a matrix of dimension (m,n) where the columns correspong to the French embeddings.
# @param R a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
#
# @return L a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
def compute_loss(X, Y, R):
    m = X.shape[0]
    diff = np.dot(X, R) - Y
    loss = (1.0/m)*np.sum(np.square(diff))
    
    return loss


## Compute Gradient Descent of least square loss function
# 
# @param X a matrix of dimension (m,n) where the columns are the English embeddings.
# @param Y a matrix of dimension (m,n) where the columns correspong to the French embeddings.
# @param R a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
# 
# @return g: a matrix of dimension (n,n) - gradient of the loss function L for given X, Y and R.
def compute_gradient(X, Y, R):
    m = X.shape[0]
    gradient = (2/m)*np.dot(X.T, np.dot(X, R) - Y)
    return gradient

## Translation Transformation matrix.
#
# @param X a matrix of dimension (m,n) where the columns are the English embeddings.
# @param Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
# @param train_steps: positive int - describes how many steps will gradient descent algorithm do.
# @param learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
# @return R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||X R -Y||^2
def align_embeddings(X, Y, train_steps=100, learning_rate=0.0003):
    np.random.seed(129)
    # the number of columns in X is the number of dimensions for a word vector (e.g. 300)
    # R is a square matrix with length equal to the number of dimensions in th  word embedding
    R = np.random.rand(X.shape[1], X.shape[1])
    
    for i in range(train_steps):
        if i % 25 == 0:
            print(f"loss at iteration {i} is: {compute_loss(X, Y, R):.4f}")
        # use the function that you defined to compute the gradient
        gradient = compute_gradient(X, Y, R)

        # update R by subtracting the learning rate times gradient
        R -= learning_rate*gradient
    return R

## K-NN algorithm
#
# @param v the vector you are going find the nearest neighbor for
# @param candidates a set of vectors where we will find the neighbors
# @param k top k nearest neighbors to find
# @param k_idx the indices of the top k closest vectors in sorted form
def nearest_neighbor(v, candidates, k=1):
    similarity_l = []
    for row in candidates:
        cos_similarity = cosine_similarity(v, row.T)
        similarity_l.append(cos_similarity)
    
    sorted_ids = np.argsort(similarity_l,0)

    # get the indices of the k most similar candidate vectors
    k_idx = sorted_ids[-k:]
    return k_idx

## Create a hash for a vector; hash_id says which random hash to use.
#
# @param v  vector of tweet. It's dimension is (1, N_DIMS)
# @param planes matrix of dimension (N_DIMS, N_PLANES) - the set of planes that divide up the region
# @param res a number which is used as a hash for your vector
def hash_value_of_vector(v, planes):
    # for the set of planes,
    # calculate the dot product between the vector and the matrix containing the planes
    # remember that planes has shape (300, 10)
    # The dot product will have the shape (1,10)
    dot_product = np.dot(v, planes)

    # get the sign of the dot product (1,10) shaped vector
    sign_of_dot_product = np.sign(dot_product)

    # set h to be false (eqivalent to 0 when used in operations) if the sign is negative,
    # and true (equivalent to 1) if the sign is positive (1,10) shaped vector
    h = [1 if s>=0 else 0 for s in sign_of_dot_product.reshape(planes.shape[1])]
    # remove extra un-used dimensions (convert this from a 2D to a 1D array)
    h = np.squeeze(h)

    # initialize the hash value to 0
    hash_value = 0

    n_planes = planes.shape[1]
    for i in range(n_planes):
        # increment the hash value by 2^i * h_i
        hash_value += 2**i * h[i]

    # cast hash_value as an integer
    hash_value = int(hash_value)
    
    return hash_value

    """
    Input:
        - vecs: list of vectors to be hashed.
        - planes: the matrix of planes in a single "universe", with shape (embedding dimensions, number of planes).
    Output:
        - hash_table: dictionary - keys are hashes, values are lists of vectors (hash buckets)
        - id_table: dictionary - keys are hashes, values are list of vectors id's
                            (it's used to know which tweet corresponds to the hashed vector)
    """
def make_hash_table(vecs, planes):

    # number of planes is the number of columns in the planes matrix
    num_of_planes = planes.shape[1]

    # number of buckets is 2^(number of planes)
    num_buckets = 2**num_of_planes

    # create the hash table as a dictionary.
    # Keys are integers (0,1,2.. number of buckets)
    # Values are empty lists
    hash_table = {i:[] for i in range(num_buckets)}

    # create the id table as a dictionary.
    # Keys are integers (0,1,2... number of buckets)
    # Values are empty lists
    id_table = {i:[] for i in range(num_buckets)}

    # for each vector in 'vecs'
    for i, v in enumerate(vecs):
        # calculate the hash value for the vector
        h = hash_value_of_vector(v, planes)

        # store the vector into hash_table at key h,
        # by appending the vector v to the list at key h
        hash_table[h]+=[v]

        # store the vector's index 'i' (each document is given a unique integer 0,1,2...)
        # the key is the h, and the 'i' is appended to the list at key h
        id_table[h]+=[i]

    return hash_table, id_table


##Search for k-NN using hashes.
def approximate_knn(doc_id, v, planes_l, k=1, num_universes_to_use=N_UNIVERSES):
    
    assert num_universes_to_use <= N_UNIVERSES

    # Vectors that will be checked as possible nearest neighbor
    vecs_to_consider_l = list()

    # list of document IDs
    ids_to_consider_l = list()

    # create a set for ids to consider, for faster checking if a document ID already exists in the set
    ids_to_consider_set = set()

    # loop through the universes of planes
    for universe_id in range(num_universes_to_use):

        # get the set of planes from the planes_l list, for this particular universe_id
        planes = planes_l[universe_id]

        # get the hash value of the vector for this set of planes
        hash_value = hash_value_of_vector(v, planes)

        # get the hash table for this particular universe_id
        hash_table = hash_tables[universe_id]

        # get the list of document vectors for this hash table, where the key is the hash_value
        document_vectors_l = hash_table[hash_value]

        # get the id_table for this particular universe_id
        id_table = id_tables[universe_id]

        # get the subset of documents to consider as nearest neighbors from this id_table dictionary
        new_ids_to_consider = id_table[hash_value]

        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

        # remove the id of the document that we're searching
        if doc_id in new_ids_to_consider:
            new_ids_to_consider.remove(doc_id)
            print(f"removed doc_id {doc_id} of input vector from new_ids_to_search")

        # loop through the subset of document vectors to consider
        for i, new_id in enumerate(new_ids_to_consider):

            # if the document ID is not yet in the set ids_to_consider...
            if new_id not in ids_to_consider_set:
                # access document_vectors_l list at index i to get the embedding
                # then append it to the list of vectors to consider as possible nearest neighbors
                document_vector_at_i = document_vectors_l[i]
                vecs_to_consider_l.append(document_vector_at_i)

                # append the new_id (the index for the document) to the list of ids to consider
                ids_to_consider_l.append(new_id)

                # also add the new_id to the set of ids to consider
                # (use this to check if new_id is not already in the IDs to consider)
                #temp=[item for item in ids_to_consider_set]
                #temp+=[new_id]
                #ids_to_consider_set=set(i for i in temp)
                ids_to_consider_set.add(new_id)

    # Now run k-NN on the smaller set of vecs-to-consider.
    print("Fast considering %d vecs" % len(vecs_to_consider_l))

    # convert the vecs to consider set to a list, then to a numpy array
    vecs_to_consider_arr = np.array(vecs_to_consider_l)

    # call nearest neighbors on the reduced list of candidate vectors
    nearest_neighbor_idx_l = nearest_neighbor(v, vecs_to_consider_arr, k=k)

    # Use the nearest neighbor index list as indices into the ids to consider
    # create a list of nearest neighbors by the document ids
    nearest_neighbor_ids = [ids_to_consider_l[idx]
                            for idx in nearest_neighbor_idx_l]

    return nearest_neighbor_ids
