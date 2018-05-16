#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot
from tester import dump_classifier_and_data, test_classifier
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments','total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'other', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    if(val > 600000 or val < 1000):
        outliers.append((key,int(val)))

        salary = data_dict[key]['salary']
        bonus = data_dict[key]['bonus']
        matplotlib.pyplot.scatter( salary, bonus) 

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


for x in outliers:
    #print(x)
    data_dict.pop(x[0],0)


### Task 3: Create new feature(s)
    
my_dataset = data_dict
def computeFraction(poi_messages, all_messages ):
    fraction = 0.
    if poi_messages == 'NaN' or all_messages == 'NaN':
        return 0.;
    else:
        fraction = float(poi_messages)/float(all_messages)
        return fraction

for i in data_dict:
    frac = computeFraction(data_dict[i]['from_poi_to_this_person'] , data_dict[i]['to_messages'])
    data_dict[i]['fraction_from_poi'] = frac
    poi_msg = data_dict[i]['from_this_person_to_poi']
    all_msg = data_dict[i]['from_messages']
    fract = computeFraction(poi_msg , all_msg)
    data_dict[i]['fraction_to_poi'] = fract
    

my_dataset = data_dict
features_list = ['poi','shared_receipt_with_poi','total_stock_value','fraction_to_poi','fraction_from_poi','expenses','bonus']


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list,remove_NaN=True, sort_keys = False)
labels, features = targetFeatureSplit(data)

### Select k best features
k=4
k_best = SelectKBest(k=k)
k_best.fit(features, labels)
scores = k_best.scores_
#print(scores)

features_list=['poi','shared_receipt_with_poi','total_stock_value','fraction_to_poi','bonus']

data = featureFormat(my_dataset, features_list,remove_NaN=True, sort_keys = False)
labels, features = targetFeatureSplit(data)

### Feature Scaling
for i in range(len(features_list)-1):
    tmp =[]
    k=0
    for x in features:
        tmp.append(float(x[i]))
    tmp = MinMaxScaler().fit_transform(tmp)
    for x in features:
        x[i]=tmp[k]
        k = k + 1


### Task 4: Try a varity of classifiers

clf =  DecisionTreeClassifier()
'''
clf= GaussianNB()
clf= SVC()
clf= RandomForestClassifier()

'''
test_classifier(clf,my_dataset,features_list)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# CHECKING BEST CLASSIFIER

# DECISION TREES

clf= DecisionTreeClassifier()
tree_para={'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],'min_samples_split':[2,3,4,5,8,10,12,15,20,25,30,35,40]}
clf = GridSearchCV(clf,tree_para)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

# RANDOM FOREST
'''
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
tree_para={'n_estimators':[70,80,90,100],'max_depth':[4,5,6,7,8,9,10],'min_samples_split':[2,3,4,5]}
clf = GridSearchCV(clf,tree_para)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
'''

# SUPPORT VECTOR MACHINE
'''
clf = SVC()
tree_para={'C':[1.0, 10.0, 100.0],'kernel':['linear', 'rbf']}
clf = GridSearchCV(clf,tree_para)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
'''

# GAUSSIAN NAIVE BAYAES
'''
clf = GaussianNB()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
'''


print("Accuracy Score:")
print (accuracy_score(pred,labels_test))
print("Recall Score:")
print(recall_score(labels_test, pred))



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

