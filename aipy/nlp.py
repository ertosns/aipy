import nltk
import numpy as np
import pandas as pd
filePath = f"~/recovery-data/AI/NLP/course/nlp/work/tmp2/"
nltk.data.path.append(filePath)
from nltk.corpus import twitter_samples, stopwords
from nlp_utils import process_tweet, build_freqs

from utils import *


##########################
#  prepare dataset
##########################

# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


# split the data into two pieces, one for training and one for testing (validation set) 
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg 
test_x = test_pos + test_neg

# combine positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# Print the shape train and test sets
print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))
'''
train_y.shape = (8000, 1)
test_y.shape = (2000, 1)
'''

##########################
#  dataset fetched!
##########################


# create frequency dictionary
freqs = build_freqs(train_x, train_y)

# check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))
'''
type(freqs) = <class 'dict'>
len(freqs) = 11346
'''

print('This is an example of a positive tweet: \n', train_x[0])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet(train_x[0]))
'''
This is an example of a positive tweet: 
 #FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)

This is an example of the processes version: 
 ['followfriday', 'top', 'engag', 'member', 'commun', 'week', ':)']
'''

## Extract positive/negative features in a given tweet
# @param tweet a list of words for one tweet
# @param freqs a dictionary corresponding to the frequencies of each tuple (word, label)
# @return x a feature vector of dimension (1,3)
def extract_pos_neg_features(tweet, freqs):
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)
    
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    x[0,0] = 1   #bias term is set to 1
    for word in word_l:
        # increment the word count for the positive label 1
        x[0,1] += freqs.get((word, 1), 0)
        # increment the word count for the negative label 0
        x[0,2] += freqs.get((word, 0), 0)
        
    assert(x.shape == (1, 3))
    return x

tmp1 = extract_pos_neg_features(train_x[0], freqs)
print(tmp1)
'''
[[1.00e+00 3.02e+03 6.10e+01]]
'''
tmp2 = extract_pos_neg_features('blorb bleeeeb bloooob', freqs)
print(tmp2)
'''
[[1. 0. 0.]]
'''

# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
# X (m,3)
for i in range(len(train_x)):
    X[i, :]= extract_pos_neg_features(train_x[i], freqs)
# training labels corresponding to X
Y = train_y

# Apply gradient descent
w=np.zeros((3, 1))
b=np.zeros((3, 1))
J, theta = gradient_descent(w, b, X, Y, 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")
'''
The cost after training is 0.24216529.
The resulting vector of weights is [7e-08, 0.0005239, -0.00055517]
'''


##
# @param tweet a string tweet
# @param freqs a dictionary corresponding to the frequencies of each tuple (word, label)
# @param theta (3,1) vector of weights
# @return y_pred the probability of a tweet being positive or negative
def predict_tweet(tweet, freqs, theta):
    # extract the features of the tweet and store it into x
    x = extract_pos_neg_features(tweet, freqs)
    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred


##
# @param test_x a list of tweets
# @param test_y (m, 1) vector with the corresponding labels for the list of tweets
# @param freqs a dictionary with the frequency of each pair (or tuple)
# @param theta weight vector of dimension (3, 1)
# @return accuracy (# of tweets classified correctly) / (total # of tweets)
def test_logistic_regression(test_x, test_y, freqs, theta):
    # the list for storing predictions
    y_hat = []
    m = test_y.shape[0]
    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)
        
        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat+=[1]
        else:
            # append 0 to the list
            y_hat+=[0]
    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = np.sum((np.array(y_hat).reshape((m,1))==test_y))/test_y.shape[0]
    return accuracy

tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")

'''
0.9950
'''

###########
# c1/w2
###########

def test_lookup(func):
    freqs = {('sad', 0): 4,
             ('happy', 1): 12,
             ('oppressed', 0): 7}
    word = 'happy'
    label = 1
    if func(freqs, word, label) == 12:
        return 'SUCCESS!!'
    return 'Failed Sanity Check!'


logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
print(logprior)
print(len(loglikelihood))
'''
0.0
9089
'''

my_tweet = 'She smiled.'
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print('The expected output is', p)
'''
The expected output is 1.5740278623499175
'''

def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    accuracy = 0  # return this properly

    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0
        # append the predicted class to the list y_hats
        y_hats+=[y_hat_i]
    # error is the average of the absolute values of the differences between y_hats and test_y
    error = None
    # Accuracy is 1 minus the error
    accuracy = sum((y_hats==test_y)>0)/float(len(test_y))

    return accuracy
print("Naive Bayes accuracy = %0.4f" %
      (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))
'''
0.9940
'''

# Run this cell to test your function
for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    # print( '%s -> %f' % (tweet, naive_bayes_predict(tweet, logprior, loglikelihood)))
    p = naive_bayes_predict(tweet, logprior, loglikelihood)
#     print(f'{tweet} -> {p:.2f} ({p_category})')
    print(f'{tweet} -> {p:.2f}')


'''
    I am happy -> 2.15
    I am bad -> -1.29
    this movie should have been great. -> 2.14
    great -> 2.14
    great great -> 4.28
    great great great -> 6.41
    great great great great -> 8.55

'''
# Feel free to check the sentiment of your own tweet below
my_tweet = 'you are bad :('
naive_bayes_predict(my_tweet, logprior, loglikelihood)

get_ratio(freqs, 'happi')
'''
{'positive': 161, 'negative': 18, 'ratio': 8.526315789473685}
'''

get_words_by_threshold(freqs, label=0, threshold=0.05)


# Some error analysis done for you
print('Truth Predicted Tweet')
for x, y in zip(test_x, test_y):
    y_hat = naive_bayes_predict(x, logprior, loglikelihood)
    if y != (np.sign(y_hat) > 0):
        print('%d\t%0.2f\t%s' % (y, np.sign(y_hat) > 0, ' '.join(
            process_tweet(x)).encode('ascii', 'ignore')))

# Test with your own tweet - feel free to modify `my_tweet`
my_tweet = 'I am happy because I am learning :)'

p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print(p)


##########
# c1/w3
##########
import pandas as pd
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('~/recovery-data/AI/NLP/course/nlp/work/Week3/capitals.txt', delimiter=' ')
data.columns = ['city1', 'country1', 'city2', 'country2']

word_embeddings = pickle.load(open("~/recovery-data/AI/NLP/course/nlp/work/Week3/word_embeddings_subset.p", "rb"))
len(word_embeddings)  # there should be 243 words that will be used in this assignment

##
# @param embeddings a word 
# @param fr_embeddings
# @param words a list of words
# @return X a matrix where the rows are the embeddings corresponding to the rows on the list
def get_vectors(embeddings, words):
    m = len(words)
    X = np.zeros((1, 300))
    for word in words:
        english = word
        eng_emb = embeddings[english]
        X = np.row_stack((X, eng_emb))
    X = X[1:,:]
    return X

print("dimension: {}".format(word_embeddings['Spain'].shape[0]))
'''
300
'''


# feel free to try different words
king = word_embeddings['king']
queen = word_embeddings['queen']

cosine_similarity(king, queen)
'''
0.6510956
'''

# Test your function
euclidean(king, queen)
'''
2.4796925
'''

## get country of given capital with given embeddings by victor subtraction
#
# @param city1 a string (the capital city of country1)
# @param country1 a string (the country of capital1)
# @param city2 a string (the capital city of country2)
# @param embeddings a dictionary where the keys are words and values are their embeddings
# @param countries a dictionary with the most likely country and its similarity score
def get_country(city1, country1, city2, embeddings):
    # store the city1, country 1, and city 2 in a set called group
    group = set((city1, country1, city2))
    # get embeddings of city 1
    city1_emb = embeddings[city1]
    # get embedding of country 1
    country1_emb = embeddings[country1]
    # get embedding of city 2
    city2_emb = embeddings[city2]
    # get embedding of country 2 (it's a combination of the embeddings of country 1, city 1 and city 2)
    # Remember: King - Man + Woman = Queen
    vec = city2_emb - city1_emb+country1_emb
    # Initialize the similarity to -1 (it will be replaced by a similarities that are closer to +1)
    similarity = -1
    # initialize country to an empty string
    country = ''
    # loop through all words in the embeddings dictionary
    for word in embeddings.keys():
        # first check that the word is not already in the 'group'
        if word not in group:
            # get the word embedding
            word_emb = embeddings[word]
            # calculate cosine similarity between embedding of country 2 and the word in the embeddings dictionary
            cur_similarity = cosine_similarity(word_emb, vec)
            # if the cosine similarity is more similar than the previously best similarity...
            if cur_similarity > similarity:
                # update the similarity to the new, better similarity
                similarity = cur_similarity
                # store the country as a tuple, which contains the word and the similarity
                country = (word, similarity)
    return country

get_country('Athens', 'Greece', 'Cairo', word_embeddings)
'''
('Egypt', 0.7626821)
'''

    '''
    Input:
        word_embeddings: a dictionary where the key is a word and the value is its embedding
        data: a pandas dataframe containing all the country and capital city pairs
    
    Output:
        accuracy: the accuracy of the model
    '''

def get_accuracy(word_embeddings, data):
    # initialize num correct to zero
    num_correct = 0

    # loop through the rows of the dataframe
    for i, row in data.iterrows():
        # get city1
        city1 = row.city1

        # get country1
        country1 = row.country1

        # get city2
        city2 =  row.city2

        # get country2
        country2 = row.country2

        # use get_country to find the predicted country2
        predicted_country2, _ = get_country(city1, country1, city2, word_embeddings)

        # if the predicted country2 is the same as the actual country2...
        if predicted_country2 == country2:
            # increment the number of correct by 1
            num_correct += 1

    # get the number of rows in the data dataframe (length of dataframe)
    m = len(data)

    # calculate the accuracy by dividing the number correct by m
    accuracy = num_correct/float(m)

    return accuracy

accuracy = get_accuracy(word_embeddings, data)
print(f"Accuracy is {accuracy:.2f}")
'''
Accuracy is 0.92
'''

# Testing your function
np.random.seed(1)
X = np.random.rand(3, 10)
X_reduced = compute_pca(X, n_components=2)
print("Your original matrix was " + str(X.shape) + " and it became:")
print(X_reduced)
'''
Your original matrix was (3, 10) and it became:
[[ 0.43437323  0.49820384]
 [ 0.42077249 -0.50351448]
 [-0.85514571  0.00531064]]
'''

words = ['oil', 'gas', 'happy', 'sad', 'city', 'town',
         'village', 'country', 'continent', 'petroleum', 'joyful']

# given a list of words and the embeddings, it returns a matrix with all the embeddings
X = get_vectors(word_embeddings, words)

print('You have 11 words each of 300 dimensions thus X.shape is:', X.shape)

# We have done the plotting for you. Just run this cell.
result = compute_pca(X, 2)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))

plt.show()


#################
# w4
#################

import pdb
import pickle
import string

import time
import gensim

import matplotlib.pyplot as plt
import nltk
import numpy as np
import scipy
import sklearn
from gensim.models import KeyedVectors
from nltk.corpus import stopwords, twitter_samples
from nltk.tokenize import TweetTokenizer

from utils import (cosine_similarity, get_dict,
                   process_tweet)
from os import getcwd


filePath = f"~/recovery-data/AI/NLP/course/nlp/work/tmp2/"
nltk.data.path.append(filePath)

en_embeddings_subset = pickle.load(open("~/recovery-data/AI/NLP/course/nlp/work/Week4/en_embeddings.p", "rb"))
fr_embeddings_subset = pickle.load(open("~/recovery-data/AI/NLP/course/nlp/work/Week4/fr_embeddings.p", "rb"))

```python
# Use this code to download and process the full dataset on your local computer

from gensim.models import KeyedVectors

en_embeddings = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary = True)
fr_embeddings = KeyedVectors.load_word2vec_format('./wiki.multi.fr.vec')

'''
# loading the english to french dictionaries
en_fr_train = get_dict('en-fr.train.txt')
print('The length of the english to french training dictionary is', len(en_fr_train))
en_fr_test = get_dict('en-fr.test.txt')
print('The length of the english to french test dictionary is', len(en_fr_train))

english_set = set(en_embeddings.vocab)
french_set = set(fr_embeddings.vocab)
en_embeddings_subset = {}
fr_embeddings_subset = {}
french_words = set(en_fr_train.values())
for en_word in en_fr_train.keys():
    fr_word = en_fr_train[en_word]
    if fr_word in french_set and en_word in english_set:
        en_embeddings_subset[en_word] = en_embeddings[en_word]
        fr_embeddings_subset[fr_word] = fr_embeddings[fr_word]


for en_word in en_fr_test.keys():
    fr_word = en_fr_test[en_word]
    if fr_word in french_set and en_word in english_set:
        en_embeddings_subset[en_word] = en_embeddings[en_word]
        fr_embeddings_subset[fr_word] = fr_embeddings[fr_word]


pickle.dump( en_embeddings_subset, open( "en_embeddings.p", "wb" ) )
pickle.dump( fr_embeddings_subset, open( "fr_embeddings.p", "wb" ) )
'''

# loading the english to french dictionaries
en_fr_train = get_dict('en-fr.train.txt')
print('The length of the English to French training dictionary is', len(en_fr_train))
en_fr_test = get_dict('en-fr.test.txt')
print('The length of the English to French test dictionary is', len(en_fr_train))
'''
The length of the English to French training dictionary is 5000
The length of the English to French test dictionary is 5000
'''


##
# en_fr English to French dictionary
# french_vecs French words to their corresponding word embeddings.
# english_vecs English words to their corresponding word embeddings.
# X a matrix where the columns are the English embeddings.
# Y a matrix where the columns correspong to the French embeddings.
# R the projection matrix that minimizes the F norm ||X R -Y||^2.
def get_matrices(en_fr, french_vecs, english_vecs):
    # X_l and Y_l are lists of the english and french word embeddings
    X_l = list()#english_vecs.values())
    Y_l = list()#french_vecs.values())
    english_set = english_vecs.keys()
    french_set = french_vecs.keys()
    for en_word, fr_word in en_fr.items():
        if fr_word in french_set and en_word in english_set:
            en_vec = english_vecs[en_word]
            fr_vec = french_vecs[fr_word]
            X_l.append(en_vec)
            Y_l.append(fr_vec)
    X = np.stack(X_l)
    Y = np.stack(Y_l)
    return X, Y

# getting the training set:
X_train, Y_train = get_matrices(en_fr_train, fr_embeddings_subset, en_embeddings_subset)

# Testing your implementation.
np.random.seed(129)
m = 10
n = 5
X = np.random.rand(m, n)
Y = np.random.rand(m, n) * .1
R = align_embeddings(X, Y)
'''
loss at iteration 0 is: 3.7242
loss at iteration 25 is: 3.6283
loss at iteration 50 is: 3.5350
loss at iteration 75 is: 3.4442
'''

R_train = align_embeddings(X_train, Y_train, train_steps=400, learning_rate=0.8)
'''
loss at iteration 0 is: 963.0146
loss at iteration 25 is: 97.8292
loss at iteration 50 is: 26.8329
loss at iteration 75 is: 9.7893
loss at iteration 100 is: 4.3776
loss at iteration 125 is: 2.3281
loss at iteration 150 is: 1.4480
loss at iteration 175 is: 1.0338
loss at iteration 200 is: 0.8251
loss at iteration 225 is: 0.7145
loss at iteration 250 is: 0.6534
loss at iteration 275 is: 0.6185
loss at iteration 300 is: 0.5981
loss at iteration 325 is: 0.5858
loss at iteration 350 is: 0.5782
loss at iteration 375 is: 0.5735
'''

# Test your implementation:
v = np.array([1, 0, 1])
candidates = np.array([[1, 0, 5], [-2, 5, 3], [2, 0, 1], [6, -9, 5], [9, 9, 9]])
ids = nearest_neighbor(v, candidates, 3)
print(candidates[ids])
'''
[[9 9 9]
 [1 0 5]
 [2 0 1]]
'''

##
# @param X: a matrix where the columns are the English embeddings.
# @param Y a matrix where the columns correspong to the French embeddings.
# @param R the transform matrix which translates word embeddings from English to French word vector space.
# @return accuracy for the English to French capitals
def test_vocabulary(X, Y, R):
    # The prediction is X times R
    pred = np.dot(X,R)

    # initialize the number correct to zero
    num_correct = 0

    # loop through each row in pred (each transformed embedding)
    for i in range(len(pred)):
        # get the index of the nearest neighbor of pred at row 'i'; also pass in the candidates in Y
        pred_idx = nearest_neighbor(pred[i], Y, 1)

        # if the index of the nearest neighbor equals the row of i... \
        if pred_idx == i:
            # increment the number correct by 1.
            num_correct += 1

    # accuracy is the number correct divided by the number of rows in 'pred' (also number of rows in X)
    accuracy = num_correct/float(len(pred))

    return accuracy

X_val, Y_val = get_matrices(en_fr_test, fr_embeddings_subset, en_embeddings_subset)
acc = test_vocabulary(X_val, Y_val, R_train)  # this might take a minute or two
print(f"accuracy on test set is {acc:.3f}")
'''
0.557
'''

all_tweets = all_positive_tweets + all_negative_tweets


##
#        - tweet: a string
#        - en_embeddings: a dictionary of word embeddings
#        - doc_embedding: sum of all word embeddings in the tweet
def get_document_embedding(tweet, en_embeddings):     
    doc_embedding = np.zeros(300)

    # process the document into a list of words (process the tweet)
    processed_doc = process_tweet(tweet)
    for word in processed_doc:
        # add the word embedding to the running total for the document embedding
        doc_embedding += en_embeddings.get(word, 0)

    return doc_embedding

# testing your function
custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"
tweet_embedding = get_document_embedding(custom_tweet, en_embeddings_subset)
tweet_embedding[-5:]

'''
array([-0.00268555, -0.15378189, -0.55761719, -0.07216644, -0.32263184])
'''

##
# @param all_docs list of strings - all tweets in our dataset.
# @param en_embeddings: dictionary with words as the keys and their embeddings as the values.
# @return document_vec_matrix matrix of tweet embeddings.
# @return  ind2Doc_dict: dictionary with indices of tweets in vecs as keys and their embeddings as the values.
def get_document_vecs(all_docs, en_embeddings):
    ind2Doc_dict = {}
    document_vec_l = []
    for i, doc in enumerate(all_docs):

        # get the document embedding of the tweet
        doc_embedding = get_document_embedding(doc, en_embeddings)

        # save the document embedding into the ind2Tweet dictionary at index i
        ind2Doc_dict[i] = doc_embedding

        # append the document embedding to the list of document vectors
        document_vec_l.append(doc_embedding)

    document_vec_matrix = np.vstack(document_vec_l)

    return document_vec_matrix, ind2Doc_dict

document_vecs, ind2Tweet = get_document_vecs(all_tweets, en_embeddings_subset)

print(f"length of dictionary {len(ind2Tweet)}")
print(f"shape of document_vecs {document_vecs.shape}")

'''
length of dictionary 10000
shape of document_vecs (10000, 300)
'''

my_tweet = 'i am sad'
process_tweet(my_tweet)
tweet_embedding = get_document_embedding(my_tweet, en_embeddings_subset)

idx = np.argmax(cosine_similarity(document_vecs, tweet_embedding))
print(all_tweets[idx])
'''
@zoeeylim sad sad sad kid :( it's ok I help you watch the match HAHAHAHAHA
'''

N_VECS = len(all_tweets)       # This many vectors.
N_DIMS = len(ind2Tweet[1])     # Vector dimensionality.
print(f"Number of vectors is {N_VECS} and each has {N_DIMS} dimensions.")


# The number of planes. We use log2(625) to have ~16 vectors/bucket.
N_PLANES = 10
# Number of times to repeat the hashing to improve the search.
N_UNIVERSES = 25

np.random.seed(0)
planes_l = [np.random.normal(size=(N_DIMS, N_PLANES))
            for _ in range(N_UNIVERSES)]




np.random.seed(0)
idx = 0
planes = planes_l[idx]  # get one 'universe' of planes to test the function
vec = np.random.rand(1, 300)
print(f" The hash value for this vector,",
      f"and the set of planes at index {idx},",
      f"is {hash_value_of_vector(vec, planes)}")
'''
 The hash value for this vector, and the set of planes at index 0, is 768
'''


np.random.seed(0)
planes = planes_l[0]  # get one 'universe' of planes to test the function
vec = np.random.rand(1, 300)
tmp_hash_table, tmp_id_table = make_hash_table(document_vecs, planes)

print(f"The hash table at key 0 has {len(tmp_hash_table[0])} document vectors")
print(f"The id table at key 0 has {len(tmp_id_table[0])}")
print(f"The first 5 document indices stored at key 0 of are {tmp_id_table[0][0:5]}")

'''
The hash table at key 0 has 3 document vectors
The id table at key 0 has 3
The first 5 document indices stored at key 0 of are [3276, 3281, 3282]
'''

hash_tables = []
id_tables = []
for universe_id in range(N_UNIVERSES):  # there are 25 hashes
    print('working on hash universe #:', universe_id)
    planes = planes_l[universe_id]
    hash_table, id_table = make_hash_table(document_vecs, planes)
    hash_tables.append(hash_table)
    id_tables.append(id_table)


#document_vecs, ind2Tweet
doc_id = 0
doc_to_search = all_tweets[doc_id]
vec_to_search = document_vecs[doc_id]

# UNQ_C22 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# You do not have to input any code in this cell, but it is relevant to grading, so please do not change anything

# Sample
nearest_neighbor_ids = approximate_knn(
    doc_id, vec_to_search, planes_l, k=3, num_universes_to_use=5)

'''
removed doc_id 0 of input vector from new_ids_to_search
removed doc_id 0 of input vector from new_ids_to_search
removed doc_id 0 of input vector from new_ids_to_search
removed doc_id 0 of input vector from new_ids_to_search
removed doc_id 0 of input vector from new_ids_to_search
Fast considering 77 vecs
'''

print(f"Nearest neighbors for document {doc_id}")
print(f"Document contents: {doc_to_search}")
print("")

for neighbor_id in nearest_neighbor_ids:
    print(f"Nearest neighbor at document id {neighbor_id}")
    print(f"document contents: {all_tweets[neighbor_id]}")

'''
Nearest neighbors for document 0
Document contents: #FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)

Nearest neighbor at document id 2140
document contents: @PopsRamjet come one, every now and then is not so bad :)
Nearest neighbor at document id 701
document contents: With the top cutie of Bohol :) https://t.co/Jh7F6U46UB
Nearest neighbor at document id 51
document contents: #FollowFriday @France_Espana @reglisse_menthe @CCI_inter for being top engaged members in my community this week :)
'''

#################
#    c2/w1
#################

from collections import Counter # collections library; counter: dict subclass for counting hashable objects

# the tiny corpus of text ! 
text = 'red pink pink blue blue yellow ORANGE BLUE BLUE PINK' # 🌈
print(text)
print('string length : ',len(text))

# convert all letters to lower case
text_lowercase = text.lower()
print(text_lowercase)
print('string length : ',len(text_lowercase))

# some regex to tokenize the string to words and return them in a list
words = re.findall(r'\w+', text_lowercase)
print(words)
print('count : ',len(words))

# create vocab
vocab = set(words)
print(vocab)
print('count : ',len(vocab))


# create vocab including word count
counts_a = dict()
for w in words:
    counts_a[w] = counts_a.get(w,0)+1
print(counts_a)
print('count : ',len(counts_a))

# create vocab including word count using collections.Counter
counts_b = dict()
counts_b = Counter(words)
print(counts_b)
print('count : ',len(counts_b))

# barchart of sorted word counts
d = {'blue': counts_b['blue'], 'pink': counts_b['pink'], 'red': counts_b['red'], 'yellow': counts_b['yellow'], 'orange': counts_b['orange']}
plt.bar(range(len(d)), list(d.values()), align='center', color=d.keys())
_ = plt.xticks(range(len(d)), list(d.keys()))


print('counts_b : ', counts_b)
print('count : ', len(counts_b))


##Candidates from String Edits
# data
word = 'dearz' # 🦌

# splits with a loop
splits_a = []
for i in range(len(word)+1):
    splits_a.append([word[:i],word[i:]])

for i in splits_a:
    print(i)

# same splits, done using a list comprehension
splits_b = [(word[:i], word[i:]) for i in range(len(word) + 1)]

for i in splits_b:
    print(i)

# deletes with a loop
splits = splits_a
deletes = []

print('word : ', word)
for L,R in splits:
    if R:
        print(L + R[1:], ' <-- delete ', R[0])

# breaking it down
print('word : ', word)
one_split = splits[0]
print('first item from the splits list : ', one_split)
L = one_split[0]
R = one_split[1]
print('L : ', L)
print('R : ', R)
print('*** now implicit delete by excluding the leading letter ***')
print('L + R[1:] : ',L + R[1:], ' <-- delete ', R[0])


# deletes with a list comprehension
splits = splits_a
deletes = [L + R[1:] for L, R in splits if R]

print(deletes)
print('*** which is the same as ***')
for i in deletes:
    print(i)

vocab = ['dean','deer','dear','fries','and','coke']
edits = list(deletes)

print('vocab : ', vocab)
print('edits : ', edits)

candidates=[]
candidates = set(vocab).intersection(edits)

print('candidate words : ', candidates)
