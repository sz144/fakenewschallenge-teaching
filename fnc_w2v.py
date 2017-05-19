# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:51:26 2017

@author: sz144
"""


import re
import numpy as np
from utils.dataset import DataSet
from utils.generate_test_splits import split
from utils.score import report_score
import time
from nltk.corpus import stopwords

import os
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format(\
'GoogleNews-vectors-negative300.bin', binary=True)


timing = time.time()
dataset = DataSet()
data_splits = split(dataset)

training_data = data_splits['training']
dev_data = data_splits['dev']
test_data = data_splits['test']
stop = set(stopwords.words('english'))


def preprocess(content):
    findword = re.compile('[a-z]+')
    temp_list = []
    content = content.lower()
    word_list = content.split()
    for word in word_list:
        new_word = findword.findall(word)
        if new_word !=[]:
            w=new_word[0].lower()
            if w not in stop and model.__contains__(w):
                temp_list.append(w)
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
    if (h != [] and b != []):# calculate the cosine of headline and body
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

train_feature_file = "features/train_features.npy"
train_label_file = "features/train_labels.npy"
dev_feature_file = "features/dev_features.npy"
dev_label_file = "features/dev_labels.npy"
test_feature_file = "features/test_features.npy"
test_label_file = "features/test_labels.npy"

# generate training data features
# load training data from file if exists
if ((not os.path.isfile(train_feature_file)) or (not os.path.isfile(train_label_file))):
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
labels=np.load(train_label_file)

print("Done in {}".format(str(time.time() - timing)))
timing = time.time()

# generate development data features
# load data from file if exists
if (not os.path.isfile(dev_feature_file)) or (not os.path.isfile(dev_label_file)):
    print("Extracting features for development data...")
    dev_X = generate_features(dev_data[0])
    dev_labels = []
    dev_labels.append(label_convert(dev_data[0]['Stance']))
    for i in range(1,len(dev_data)):
        stance = dev_data[i]
        features = generate_features(stance)
        dev_X = np.vstack((dev_X, features))
        dev_labels.append(label_convert(stance['Stance']))
    np.save(dev_feature_file, dev_X)
    np.save(dev_label_file, dev_labels)
dev_X = np.load(dev_feature_file)
dev_labels = np.load(dev_label_file)



# generate test data features
# load data from file if exists
if (not os.path.isfile(test_feature_file)) or (not os.path.isfile(test_label_file)):
    print("Extracting features for testing data...")
    test_X = generate_features(test_data[0])
    test_labels = []
    test_labels.append(test_data[0]['Stance'])
    for i in range(1,len(test_data)):
        stance = test_data[i]
        features = generate_features(stance)
        test_X = np.vstack((test_X, features))
        test_labels.append(stance['Stance'])
    np.save(test_feature_file, test_X)
    np.save(test_label_file, test_labels)
test_X = np.load(test_feature_file)
test_labels = np.load(test_label_file)

print("Done in {}".format(str(time.time() - timing)))
timing = time.time()

# training
print("Training neural network...")
nn_clf = MLPClassifier()
nn_clf.fit(X, labels)

#print("Training SVM...")
#rbf_svc = svm.SVC(kernel='rbf')
#rbf_svc.fit(X, labels)
#lr_clf = LogisticRegression()
#lr_clf.fit(X,labels)

print("Done in {}".format(str(time.time() - timing)))
timing = time.time()

print("Evaluating...")
pred = nn_clf.predict(test_X)
#pred = rbf_svc.predict(test_X)
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

report_score(test_labels,pred_labels)
print("Done in {}".format(str(time.time() - timing)))
