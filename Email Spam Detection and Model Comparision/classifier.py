import os
import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
import csv
import random
import math
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import random
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  


def build_dictionary(dir):
  # read all the files from specified folder
  emails = os.listdir(dir)
  emails.sort()
  #print(emails)
  # it contains all the mail contents
  dictionary = []

  # we are splitting all the mesgs and store the words
  for email in emails:
    m = open(os.path.join(dir, email))
    
    for i, line in enumerate(m):
      
      if i == 2: # email message only
        words = line.split()
        dictionary += words

  # it will remove the duplicates
  dictionary = list(set(dictionary))

  # deletes the unwanted data
  for index, word in enumerate(dictionary):
    if (word.isalpha() == False) or (len(word) == 1):
      del dictionary[index]
  #print(dictionary) 
  return dictionary

def build_features(dir, dictionary):
  # Read the emails
  emails = os.listdir(dir)
  emails.sort()
  # array to store features
  features_matrix = np.zeros((len(emails), len(dictionary)))

  # storing of words and its frequency
  for email_index, email in enumerate(emails):
    m = open(os.path.join(dir, email))
    for line_index, line in enumerate(m):
      if line_index == 2:
        words = line.split()
        for word_index, word in enumerate(dictionary):
          #print(word,"------",words.count(word))
          features_matrix[email_index, word_index] = words.count(word)

  return features_matrix 

def build_labels(dir):
  # Read the emails
  emails = os.listdir(dir)
  emails.sort()
  
  # array for label storage
  labels_matrix = np.zeros(len(emails))

  for index, email in enumerate(emails):
    labels_matrix[index] = 1 if re.search('spms*', email) else 0

  return labels_matrix




# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    return y_pred

  
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):

    return accuracy_score(y_test, y_pred) * 100





train_dir = './train_data'
print('1. Building dictionary object')
dictionary = build_dictionary(train_dir)

print('2. Building training features and labels for classifier')
features_train = build_features(train_dir, dictionary)
#print(features_train)
labels_train = build_labels(train_dir)
#print(labels_train)   


######################################Naive Bayes #############################################################
classifier = MultinomialNB()

print('3. Training the classifier to prediction')
classifier.fit(features_train, labels_train)


test_dir = './test_data'
print('4. Building the test features and labels')
features_test = build_features(test_dir, dictionary)
labels_test = build_labels(test_dir)

pred=classifier.predict(features_test)
print('5. Calculating accuracy of the trained classifier')

print ('Accuracy in Naive Bayes Label Data: ', (accuracy_score(labels_test,pred)*100),' %')

######################################Linear SVC #############################################################
classifier=LinearSVC()     
classifier.fit(features_train, labels_train)
pred=classifier.predict(features_test)
accuracy = classifier.score(features_test, labels_test)  
print ('Accuracy in Linear SVC Label Data: ', accuracy,' %')  

####################################Decision Tree ############################################################

clf_gini1 = DecisionTreeClassifier(criterion="gini",random_state=100, max_depth=3, min_samples_leaf=5)

#Performing training action
clf_gini1.fit(features_train, labels_train)
y_pred_gini1 = prediction(features_test, clf_gini1)
vnew3 = cal_accuracy(labels_test, y_pred_gini1)
print("Accuracy in DecisionTreeClassifier using Gini Index for Label Data: ",vnew3,' %')


####################################Logistic Regression############################################################
Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(features_train, labels_train)
pred = Spam_model.predict(features_test)
LR_accuracy=(accuracy_score(labels_test,pred)*100)
print("Accuracy in LogisticRegression using Gini Index for Label Data: ",LR_accuracy,' %')

