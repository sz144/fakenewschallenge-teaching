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
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
#from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm

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

# convert word to vector
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
    cos = np.zeros((1,1))
    if (h != [] and b != []):# calculate the cosine of headline and body
        cos[0,0] = model.n_similarity(h,b)
    if (h != []):
        h_vector = to_vector(h)
    else:
        h_vector = np.zeros((1,300))
    if (b != []):
        b_vector = to_vector(b)
    else:
        b_vector = np.zeros((1,300))
    long_vector = np.hstack((h_vector,b_vector))
    features = np.hstack((cos,long_vector))
    return features

# FNC base line features
def baseline_features(input_data,dataset,name):
    h, b = [],[]
    for i in range(len(input_data)):
        stance = input_data[i]
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X

# convert labels to numbers for training
def label_convert(label):
    if label == 'unrelated':
        return 0
    elif label == 'agree':
        return 1
    elif label == 'disagree':
        return 2
    else:
        return 3

def plot_learning_curve(estimator, title, X, y, dev_X, dev_labels):
    plt.figure()
    plt.title(title)
    best_score = 0
    best_estimator=estimator
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    size=[int(X.shape[0]*0.2),int(X.shape[0]*0.4),int(X.shape[0]*0.6),int(X.shape[0]*0.8),X.shape[0]]
    train_score=[]
    test_score=[]
    for s in size:
        sample_X=X[:s,:]
        sample_y=y[:s]
        estimator.fit(sample_X,sample_y)
        train_score.append(estimator.score(sample_X,sample_y))
        t_score=estimator.score(dev_X,dev_labels)
        test_score.append(t_score)
        if t_score>best_score:
            best_score=t_score
            best_estimator=estimator
    plt.plot(size,train_score,'bx-',label="score on training data")
    plt.plot(size,test_score,'rx-', label="score on development data")
    plt.legend(loc="best")
    plt.show()
    print("Example sizes: {}".format(size))
    print("Training score: {}".format(train_score))
    print("Test score: {}".format(test_score))
    return best_estimator

def generate_labels(pred):
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
    return pred_labels

train_feature_file = "features/train_features.npy"
train_label_file = "features/train_labels.npy"
dev_feature_file = "features/dev_features.npy"
dev_label_file = "features/dev_labels.npy"
test_feature_file = "features/test_features.npy"
test_label_file = "features/test_labels.npy"

# generate training data features
# load training data from file if exists
print("Extracting training features...")
base_X = baseline_features(training_data,dataset,'train')
if ((not os.path.isfile(train_feature_file)) or (not os.path.isfile(train_label_file))):
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
X = np.load(train_feature_file)
X = np.hstack((base_X,X))
labels = np.load(train_label_file)

print("Done in {}".format(str(time.time() - timing)))
timing = time.time()

# generate development data features
# load data from file if exists
base_dev_X=baseline_features(dev_data,dataset,'dev')
if (not os.path.isfile(dev_feature_file)) or (not os.path.isfile(dev_label_file)):
    #print("Extracting features for development data...")
    dev_X = generate_features(dev_data[0])
    dev_labels = []
    dev_labels.append(label_convert(dev_data[0]['Stance']))
    for i in range(1,len(dev_data)):
        stance = dev_data[i]
        features = generate_features(stance)
        dev_X = np.vstack((dev_X, features))
        dev_labels.append(stance['Stance'])
    np.save(dev_feature_file, dev_X)
    np.save(dev_label_file, dev_labels)
dev_X = np.load(dev_feature_file)
dev_X = np.hstack((base_dev_X,dev_X))
dev_labels = np.load(dev_label_file)


# generate test data features
# load data from file if exists
if (not os.path.isfile(test_feature_file)) or (not os.path.isfile(test_label_file)):
    #print("Extracting features for testing data...")
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
base_test_X = baseline_features(test_data,dataset,'test')
test_X = np.load(test_feature_file)
test_X = np.hstack((base_test_X,test_X))
test_labels = np.load(test_label_file)

print("Done in {}".format(str(time.time() - timing)))
timing = time.time()

dev_y = generate_labels(dev_labels)


# training
# uncomment the next part for using neural network

print("Training neural network...")
nn_clf = MLPClassifier(hidden_layer_sizes=(300, ), shuffle=False)
title = "Learning Curves (Neural Networks)"
nn_clf=plot_learning_curve(nn_clf,title,X,labels,dev_X,dev_labels)
print("Evaluating neural network...")
#test on dev data
nn_dev_pred = nn_clf.predict(dev_X)
nn_dev_pred_labels = generate_labels(nn_dev_pred)
print("Performance on development data:")
report_score(dev_y,nn_dev_pred_labels)
# test on test data
nn_pred = nn_clf.predict(test_X)
nn_pred_labels=generate_labels(nn_pred)
print("Performance on test data:")
report_score(test_labels,nn_pred_labels)

# uncomment the next part for using non-linear support vector machine, rbf kernel
'''
title = "Learning Curves (SVM, RBF kernel)"
print("Training SVM...")
rbf_svc = svm.SVC(decision_function_shape='ovr')
rbf_svc = plot_learning_curve(rbf_svc,title,X,labels,dev_X,dev_labels)
print("Evaluating svm...")
#test on dev data
svc_dev_pred = rbf_svc.predict(dev_X)
svc_dev_pred_labels=generate_labels(svc_dev_pred)
print("Performance on development data:")
report_score(dev_y,svc_dev_pred_labels)
# test on test data
svc_pred = rbf_svc.predict(test_X)
svc_pred_labels=generate_labels(svc_pred)
print("Performance on test data:")
report_score(test_labels,svc_pred_labels)
'''
# uncomment the next part for using logistic regression
'''
title="Learn Curves (Logistic Regression)"
print("Trainging LR")
lr_clf = LogisticRegression()
lr_clf = plot_learning_curve(lr_clf,title,X,labels,dev_X,dev_labels)
print("Evaluating LR...")
#test on dev data
lr_dev_pred = lr_clf.predict(dev_X)
lr_dev_pred_labels=generate_labels(lr_dev_pred)
print("Performance on development data:")
report_score(dev_y,lr_dev_pred_labels)
# test on test data
lr_pred = lr_clf.predict(test_X)
lr_pred_labels=generate_labels(lr_pred)
print("Performance on test data:")
report_score(test_labels,lr_pred_labels)
'''
print("Done in {}".format(str(time.time() - timing)))
