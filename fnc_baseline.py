# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 21:28:33 2017

@author: sz144
"""

from utils.dataset import DataSet
from utils.generate_test_splits import split
from utils.score import report_score
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm

dataset = DataSet()
data_splits = split(dataset)

training_data = data_splits['training']
dev_data = data_splits['dev']
test_data = data_splits['test']


def baseline_features(input_data,dataset,name):
    h, b = [],[] 
    for i in range(len(input_data)):
        stance = input_data[i]
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "baseline_features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "baseline_features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "baseline_features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "baseline_features/hand."+name+".npy")
    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X

def calculate_cos(bodies,headlines,tfidf_vectorizer):
    temp_body = []
    temp_headline = []
    temp_body.append(bodies[0])
    temp_headline.append(headlines[0])
    body_vec = tfidf_vectorizer.transform(temp_body).toarray()
    headline_vec = tfidf_vectorizer.transform(temp_headline).toarray()
    cos = np.dot(body_vec,headline_vec.T)
    for i in range(1,len(bodies)):
        temp_body = []
        temp_headline = []
        temp_body.append(bodies[i])
        temp_headline.append(headlines[i])
        # convert the str of current headline and body to list
        body_vec = tfidf_vectorizer.transform(temp_body).toarray()
        headline_vec = tfidf_vectorizer.transform(temp_headline).toarray()

        c = np.dot(body_vec,headline_vec.T)
        cos = np.vstack((cos,c))
    return cos

def label_convert(label):
    if label == 'unrelated':
        return 0
    elif label == 'agree':
        return 1
    elif label == 'disagree':
        return 2
    else:
        return 3

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


if __name__ == '__main__':
    
    train_cos_file = "baseline_features/train_cos.npy"
    dev_cos_file = "baseline_features/dev_cos.npy"
    test_cos_file = "baseline_features/test_cos.npy"
    print('loading data...')
    train_bodies = []
    train_headlines = []
    train_labels = []
    dev_bodies=[]
    dev_headlines=[]
    dev_labels=[]
    test_bodies=[]
    test_headlines=[]
    test_labels=[]
    for stance in training_data:
        train_headlines.append(stance['Headline']) 
        train_bodies.append(dataset.articles[stance['Body ID']])
        train_labels.append(label_convert(stance['Stance']))
    for stance in dev_data:
        dev_headlines.append(stance['Headline']) 
        dev_bodies.append(dataset.articles[stance['Body ID']])
        dev_labels.append(stance['Stance'])
    for stance in test_data:
        test_headlines.append(stance['Headline']) 
        test_bodies.append(dataset.articles[stance['Body ID']])
        test_labels.append(stance['Stance'])
    base_X = baseline_features(training_data,dataset,'train') 
    base_dev_X=baseline_features(dev_data,dataset,'dev')
    base_test_X = baseline_features(test_data,dataset,'test')
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(train_bodies)
    if not os.path.isfile(train_cos_file):
        train_cos = calculate_cos(train_bodies,train_headlines,tfidf_vectorizer)
        np.save(train_cos_file,train_cos)
    if not os.path.isfile(dev_cos_file):
        dev_cos = calculate_cos(dev_bodies,dev_headlines,tfidf_vectorizer)
        np.save(dev_cos_file,dev_cos)
    if not os.path.isfile(test_cos_file):
        test_cos = calculate_cos(test_bodies,test_headlines,tfidf_vectorizer)
        np.save(test_cos_file,test_cos)    
        
        
    train_cos=np.load(train_cos_file)
    dev_cos=np.load(dev_cos_file)
    test_cos=np.load(test_cos_file)
    X=np.hstack((base_X,train_cos))
    dev_X=np.hstack((base_dev_X,dev_cos))
    test_X=np.hstack((base_test_X,test_cos))
    
    
    clf = MLPClassifier()
    #clf = svm.SVC(decision_function_shape='ovr')
    #clf = LogisticRegression()
    
    clf.fit(X,train_labels)
    dev_pred = clf.predict(dev_X)    
    dev_pred_labels = generate_labels(dev_pred)
    test_pred = clf.predict(test_X)
    test_pred_labels = generate_labels(test_pred)
    report_score(dev_labels,dev_pred_labels)
    report_score(test_labels,test_pred_labels)

    
  
'''        
    for stance in dev_data:
        print(stance)
        print(dataset.articles[stance['Body ID']])
        print("")

    #Test data will be provided in week 10
    for stance in test_data:
        print(stance)
        print(dataset.articles[stance['Body ID']])
        print("")
'''        