import time
import pandas as pd
import numpy as np

df = pd.read_excel(r'C:/Users/User/Documents/University of Southampton/Year 3/SEM1/COMP3222 Machine learning/Coursework/df.xlsx')
#dft = pd.read_excel(r'C:/Users/User/Documents/University of Southampton/Year 3/SEM1/COMP3222 Machine learning/Coursework/dft.xlsx')

#trainingnset#######################
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(ngram_range=(1, 1),max_df=0.5, min_df=5, token_pattern=r'[^\s]+', use_idf = True, stop_words='english')
#tfidfmatrix
tfidf_training = v.fit_transform(df.tweetText_cleaned.values.astype('U'))
x_train = tfidf_training.toarray()
y_train = df.encoded_label
tfidf_testing = v.transform(df.tweetText_cleaned.values.astype('U'))
x_test = tfidf_testing.toarray()
y_test = df.encoded_label
###############################################

# from sklearn.decomposition import PCA
# pca = PCA(n_components = 0.95)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)
# print(pca.n_components_)
############################################

#logistic regression#
start = time.time()
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state= 42)
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)
end = time.time()
# #####################################################

#SVM#
# start = time.time()
# from sklearn import svm
# clf = svm.SVC(random_state= 42, C= 1, kernel='rbf')
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# end = time.time()
#####################

#decision tree#
# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
# start = time.time()
# tree_clf = tree.DecisionTreeClassifier(random_state=42,)
# tree_clf.fit(x_train, y_train)
# y_pred = tree_clf.predict(x_test)
# end = time.time()
#############################################

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
# https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
def get_confusion_matrix_values(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])

TP, FP, FN, TN = get_confusion_matrix_values(y_test, y_pred)
print(TP, FP, FN, TN)

#metrics calculation
precision = TP/(TP+FP)
print("precision",precision)
recall = TP/(TP+FN)
print("recall:",recall)
F1_score = 2*(precision*recall)/(precision+recall)
print("f1score:",F1_score)
print("f1score-micro:",f1_score(y_test, y_pred, average="micro"))
print(f"Runtime of the program is {end - start}")
# ###########################################################
