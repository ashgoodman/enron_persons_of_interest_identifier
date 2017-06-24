from tester import dump_classifier_and_data,test_classifier
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import os
import re
import sys
import random
import numpy
import matplotlib.pyplot as plt
import pickle
import warnings
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

def classify(dataset, feature_set, features_train, labels_train, features_test, labels_test, clf, params=None, score_method=None):   
    clf = clf
    print "classifier:", clf
    if params == None:
        parameters = {}
    else:
        parameters = params
    if score_method == None:
        clf = GridSearchCV(clf, parameters)
    else:
        clf = GridSearchCV(clf, parameters,  scoring=score_method)
    clf = clf.fit(features_train, labels_train)
    result = test_classifier(clf, dataset, feature_set)
    print "best score: ", clf.best_score_
    print "best params:", clf.best_params_
    print "scorer:", clf.scorer_
    y_true, y_pred = labels_test, clf.predict(features_test)
    print "Classification Report: ", classification_report(y_true, y_pred)
    return clf, clf.best_params_ , result

## Scaling features
## functions will scale features  from their original values to values between 0 and 1, leaving features after the stop value as is.
## feature 0 is poi, a True/False value

## I could have used sklearn's MinMaxScaler for this during training/testing of the classifiers
## but I wanted to practice implementing the code and concepts by hand to better understand the process

# funtion to get the max/min values for a given feature
def get_max_min(feature_array):
    maxval = max(feature_array)
    minval = min(feature_array)
    return maxval, minval

# function to create an array for a given feature
def create_value_array(data, feature):
    value_array = []
    for name in data:
        if data[name][feature] != 'NaN':
            value_array.append(data[name][feature])
        else:
            #Important: if NaN values are set to 0.0 after this step instead of during this step they become equal to 
            # the lowest value, for example if the lowest bonus is 50,000 scaling would make it 0.0 and then any
            # NaN would be 0.0 effectively making NaN values equal to the lowest actual value.
            
            # This becomes especially important on features like exercised_stock_options or total_stock_value
            # where the values are quite high to begin with, equating NaN to the lowest value would effectively give
            # all people in the database the same minimum value for stocks and negatively affect the determination value
            # of that feature, skewing the data by giving equal weights to NaN's and the lowest actual values.
            
            # By adding NaN values as 0.0 now I make 0.0 the lowest value already therefore the lowest actual
            # value will be somewhat higher than 0.0 while the max will still be 1.0 and the zero value of NaN's will be preserved
            value_array.append(0.0)
    return value_array    

# actual scaling function using the 2 functions above
def feature_scaling(data, features, stop_value):
    fcount = 1
    while fcount < stop_value:
        feature = features[fcount]
        values = create_value_array(data, feature)
        maxmin = get_max_min(values)
        for name in data:
            if data[name][features[fcount]] != 'NaN':
                val = data[name][features[fcount]]
                new_val = round((float(val)-float(maxmin[1]))/(float(maxmin[0])-float(maxmin[1])),5)
                data[name][features[fcount]] = new_val
            else: # may be able to remove this step as feature format takes care of this?
                data[name][features[fcount]] = 0.0
        fcount+=1
    return data



## Function to get a list of each type of person (poi vs non poi) in the dataset
def people_lists(data):
    poi_list = []
    no_poi = []
    for i in data:
        if data[i]['poi'] == True:
           poi_list.append(i)
        else:
            no_poi.append(i)

    return poi_list, no_poi

# function to show percent of peeople in dataset that do not have a given feature, by type of person (poi or non poi)
def nanList(people, features, data):
    nans ={}
    for i in features:
        nan_list = []
        for j in people:
            if data[j][i] == 'NaN':
                nan_list.append(j)
                
        nans[i]= nan_list

    for key in nans:
        print key, round(float(len(nans[key]))/float(len(people)),2)


## This function lets me look at people who are missing values equal to or above any chosen threshold using any feature set
def all_nan(data, features, num_missing_values):
    persons={}
    for i in data:
        count= 0
        feats = []
        for j in features:
                if data[i][j] == 'NaN':
                    count+=1
                    feats.append(j)
        persons[i] = count, feats
    for z in persons:
        if persons[z][0] >= num_missing_values:
            print z, ":", persons[z]


## The following function will examine this by summing feature values and comparing them to a given total value
def totals_match(data, features, num_feat_total):
    for i in data:
        values=[]
        if data[i][features[num_feat_total]] !="NaN":
            total = data[i][features[num_feat_total]]
        else:
            total = 0
        for j in features:
            if data[i][j] == "NaN":
                values.append(0)
            else:
                values.append(data[i][j])
        if sum(values)/2 != total:
            print i, sum(values)/2, total


### outlier examination function, colors points by poi status and labels by name
def outlier_examination(feature1, feature2, data):
    # create lists
    feature_1 = []
    feature_2 = []
    names=[]
    poi_status=[]
    # populate lists
    for i in data:
        if data[i][feature1] == 'NaN':
            feature_1.append(0.0), names.append(i), poi_status.append(data[i]['poi'])
        else: feature_1.append(data[i][feature1]), names.append(i), poi_status.append(data[i]['poi'])      

        if data[i][feature2] == 'NaN':
            feature_2.append(0.0)
        else: feature_2.append(data[i][feature2])

    ## Create plot
    fig, ax = plt.subplots()
    ## title and labels
    ax.set_title(feature1+' vs '+feature2)
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ## set colors
    use_colours = {True: "red", False: "green"}
    ## plot data
    ax.scatter(feature_1, feature_2, c=[use_colours[x] for x in poi_status])
    ## annotate data
    for i, txt in enumerate(names):
        ax.annotate(txt, (feature_1[i],feature_2[i]))
    plt.show()


## Creates new features percent_to_poi and percent_from_poi usinjg from amd to poi email features
def percentage_comms_poi(data):
    # Add two new features called 'percent_to_poi' and 'percent_from_poi'
    for name in data:
        if data[name]["from_poi_to_this_person"] == 'NaN' and data[name]["to_messages"] == 'NaN':
            data[name]['percent_from_poi']= 0.0
        elif data[name]["from_poi_to_this_person"] == 'NaN' and data[name]["to_messages"] != 'NaN':
            data[name]['percent_from_poi']= 0.0
        elif data[name]["from_poi_to_this_person"] != 'NaN' and data[name]["to_messages"] == 'NaN':
            data[name]['percent_from_poi']= 1.0
        else:
            data[name]['percent_from_poi']= round(float(data[name]["from_poi_to_this_person"])/float(data[name]["to_messages"]), 5)
        #print name, "percent from" , data[name]['percent_from_poi']
        
        if data[name]["from_this_person_to_poi"] == 'NaN' and data[name]["from_messages"] == 'NaN':
            data[name]['percent_to_poi']= 0.0
        elif data[name]["from_this_person_to_poi"] == 'NaN' and data[name]["from_messages"] != 'NaN':
            data[name]['percent_to_poi']= 0.0
        elif data[name]["from_this_person_to_poi"] != 'NaN' and data[name]["from_messages"] == 'NaN':
            data[name]['percent_to_poi']= 1.0
        else:
            data[name]['percent_to_poi']= round(float(data[name]["from_this_person_to_poi"])/float(data[name]["from_messages"]), 5)
        #print name, "percent to" , data[name]['percent_to_poi']
    return data


## Sanity check: checks frequency of communication between people at a given freqency threshhold
def poi_freq_sanity_check(data, poi_status, freq):
    toPoi=0
    fromPoi=0
    total = 0
    for name in data:
        if data[name]['poi'] == poi_status:
            total+=1
            if data[name]['percent_to_poi'] > freq:
                toPoi+=1
            if data[name]['percent_from_poi'] > freq:
                fromPoi+=1
    percent_to = round(float(toPoi)/float(total), 3)
    percent_from = round(float(fromPoi)/float(total), 3)
    print percent_to, percent_from, toPoi, fromPoi, total




## valid options for scoring method in GridSearchCV:
##['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples',
## 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'median_absolute_error',
## 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2',
## 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']