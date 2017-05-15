# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 21:28:33 2017

@author: sz144
"""

import re
import numpy as np
from utils.dataset import DataSet
from utils.generate_test_splits import split
from utils.score import report_score
import time
import os
#from sklearn.linear_model import LogisticRegression
#from sklearn import svm
#from utils.generate_test_splits import kfold_split, get_stances_for_folds
#from utils.score import report_score, LABELS, score_submission
#from utils.system import parse_params, check_version
from sklearn.neural_network import MLPClassifier
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format(\
'GoogleNews-vectors-negative300.bin', binary=True)


timing = time.time()
dataset = DataSet()
data_splits = split(dataset)

training_data = data_splits['training']
dev_data = data_splits['dev']
test_data = data_splits['test']

def preprocess(content):
    findword = re.compile('[a-z]+')
    temp_list = []
    content = content.lower()
    word_list = content.split()
    for word in word_list:
        new_word = findword.findall(word)
        if new_word != [] and model.__contains__(new_word[0]):
            temp_list.append(new_word[0])
    return temp_list


def to_vector(word_list):
    n = 0
    words_vector = np.zeros((1,300))
    for word in word_list:
        words_vector = words_vector + model[word].reshape((1,300))
        n += 1
    return words_vector/n


def generate_features(stance):
    h = preprocess(stance['Headline'])
    b = preprocess(dataset.articles[stance['Body ID']])
    s = np.zeros((1,1))
    if (h != [] and b != []):
        s[0,0] = model.n_similarity(h,b)
    if (h != []):
        h_vector = to_vector(h)
    else:
        h_vector = np.zeros((1,300))
    if (b != []):    
        b_vector = to_vector(b)
    else:
        b_vector = np.zeros((1,300))
    long_vector = np.hstack((h_vector,b_vector))
    features = np.hstack((s,long_vector))
    return features


def label_convert(label):
    if label == 'unrelated':
        return 0
    elif label == 'agree':
        return 1
    elif label == 'disagree':
        return 2
    else:
        return 3

train_feature_file = 'features\train_features.npy'
train_label_file = 'features\train_labels.npy'
dev_feature_file = 'features\dev_features.npy'
dev_label_file = 'features\dev_labels.npy'

# generate training data
# load training data from file if exists
if not os.path.isfile(train_feature_file) or not os.path.isfile(train_label_file):
    print("Extracting features for training data...")
    X = generate_features(training_data[0])
    labels = []
    label = training_data[0]['Stance']
    labels.append(label_convert(label))
    for i in range(1,len(training_data)):
        stance = training_data[i]
        features = generate_features(stance)
        X = np.vstack((X,features))
        labels.append(label_convert(stance['Stance']))
    np.save(train_feature_file, X)
    np.save(train_label_file, labels)
X=np.load(train_feature_file)
label=np.load(train_label_file)
    
print("Done in {}".format(str(time.time() - timing)))
timing = time.time()

# generate testing data
# load data from file if exists
if not os.path.isfile(dev_feature_file) or not os.path.isfile(dev_label_file):
    print("Extracting features for testing data...")    
    dev_X = generate_features(dev_data[0])
    dev_labels = []
    dev_labels.append(dev_data[0]['Stance'])
    for i in range(1,len(dev_data)):
        stance = dev_data[i]
        features = generate_features(stance)
        dev_X = np.vstack((dev_X, features))
        dev_labels.append(stance['Stance'])
    np.save(dev_feature_file, dev_X)
    np.save(dev_label_file, dev_labels)
dev_X = np.load(dev_feature_file)
dev_labels = np.load(dev_label_file)


print("Done in {}".format(str(time.time() - timing)))
timing = time.time()

# training 
print("Training neural network...")
nn_clf = MLPClassifier(hidden_layer_sizes=(300,4))
nn_clf.fit(X, labels)
print("Done in {}".format(str(time.time() - timing)))
timing = time.time()
print("Evaluating...")
pred = nn_clf.predict(dev_X)
pred_labels = []
for p in pred:
    if p == 0:
        pred_labels.append('unrelated')
    elif p == 1:
        pred_labels.append('agree')
    elif p == 2:
        pred_labels.append('disagree')
    else:
        pred_labels.append('discuss')

report_score(dev_labels,pred_labels)  
print("Done in {}".format(str(time.time() - timing)))    