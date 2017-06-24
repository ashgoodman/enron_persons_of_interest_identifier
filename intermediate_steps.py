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
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from intermediate_code import classify, feature_scaling, get_max_min, create_value_array, people_lists, \
 nanList, all_nan, totals_match, outlier_examination, percentage_comms_poi, poi_freq_sanity_check
from tester import dump_classifier_and_data,test_classifier
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import  SelectKBest
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



all_features_list = ['poi','salary', 'to_messages', 'total_payments','deferral_payments',
                 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',
                 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances',
                 'from_messages', 'other', 'from_this_person_to_poi','director_fees',
                 'deferred_income','long_term_incentive','from_poi_to_this_person', 'email_address']

all_finance_features_list = ['salary', 'total_payments','deferral_payments',
                 'exercised_stock_options', 'bonus', 'restricted_stock', 
                 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances',
                 'other', 'director_fees','deferred_income','long_term_incentive']

all_payment_features = ['salary', 'bonus', 'long_term_incentive',  
                 'deferred_income', 'deferral_payments', 'loan_advances','other', 
                 'expenses','director_fees','total_payments']


all_stock_features = ['exercised_stock_options', 'restricted_stock', 
                 'restricted_stock_deferred', 'total_stock_value']


all_email_features_list = ['to_messages', 'shared_receipt_with_poi',
                 'from_messages', 'from_this_person_to_poi','from_poi_to_this_person', 'email_address']



## Note email data is meta data, not actual email content

### Load the dictionary containing the dataset
def load_data(data):
    with open(data, "r") as data_file:
        data_dict = pickle.load(data_file)
        return data_dict

data_dict = load_data("final_project_dataset.pkl")



## get the length of each list and total dataset
print len(people_lists(data_dict)[0])
## 18
print len(people_lists(data_dict)[1])
##128
print len(data_dict)
##146


## We have 18 POI's and 128 non-POI's out of a total dataset of 146 people with 20 features and 1 label (poi)

## I am excluding email addresses as they are not numerical values and have no predictive power for this dataset to indicate poi vs non poi            

## I want to look at POI's and see if there are fields which are blank (NaN) for the majority/all of them.


temp_feature_list_1 = ['poi','salary', 'total_payments','to_messages','deferral_payments',
                 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',
                 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances',
                 'from_messages', 'other', 'from_this_person_to_poi','director_fees',
                 'deferred_income','long_term_incentive','from_poi_to_this_person'] 



## percent of poi who do not have a feature    
nanList(people_lists(data_dict)[0],temp_feature_list_1, data_dict)

# salary 0.06
# to_messages 0.22
# deferral_payments 0.72
# total_payments 0.0
# loan_advances 0.94
# bonus 0.11
# restricted_stock_deferred 1.0
# total_stock_value 0.0
# shared_receipt_with_poi 0.22
# long_term_incentive 0.33
# exercised_stock_options 0.33
# from_messages 0.22
# other 0.0
# from_poi_to_this_person 0.22
# from_this_person_to_poi 0.22
# poi 0.0
# deferred_income 0.39
# expenses 0.0
# restricted_stock 0.06
# director_fees 1.0

## percent of non- poi who do not have a feature
nanList(people_lists(data_dict)[1],temp_feature_list_1, data_dict)

# salary 0.39
# to_messages 0.44
# deferral_payments 0.73
# total_payments 0.16
# loan_advances 0.98
# bonus 0.48
# restricted_stock_deferred 0.86
# total_stock_value 0.16
# shared_receipt_with_poi 0.44
# long_term_incentive 0.58
# exercised_stock_options 0.3
# from_messages 0.44
# other 0.41
# from_poi_to_this_person 0.44
# from_this_person_to_poi 0.44
# poi 0.0
# deferred_income 0.7
# expenses 0.4
# restricted_stock 0.27
# director_fees 0.87


## outlier removal 
## Removing TOTAL and THE TRAVEL AGENCY IN THE PARK
data_dict.pop( "TOTAL", 0 )
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

## Next lets see how many people have empty values for all features
## Looking for anyone missing all financial data
all_nan(data_dict, all_finance_features_list, 14)
## Looking for anyone missing all email data
aall_nan(data_dict, all_email_features_list, 5)
## LOCKHART EUGENE E  is missing all financial data and email data and is not a poi, so I will remove him as an outlier
data_dict.pop("LOCKHART EUGENE E", 0)
## KOPPER MICHAEL J, FASTOW ANDREW S, YEAGER F SCOTT and HIRKO JOSEPH are all poi's who have no email meta data. As they are poi's I will retain them, 
## but this brings into question the usefullness of email meta data when almost 28% of poi's have no email meta data associated with them

## after round 1 outlier removal

##Now lets run the peoples list code again to see how many values we now have:
# get the length of each list and total dataset
print len(people_lists(data_dict)[0])
# 18
print len(people_lists(data_dict)[1])
#126
print len(data_dict)
#144


## Data examination: are all financial values correct?
## to test this all payment info must add up to total payments value and all stock info must add up to total stock values


all_payment_features = ['salary', 'bonus', 'long_term_incentive',  
                 'deferred_income', 'deferral_payments', 'loan_advances','other', 
                 'expenses','director_fees','total_payments']


all_stock_features = ['exercised_stock_options', 'restricted_stock', 
                 'restricted_stock_deferred', 'total_stock_value']



totals_match(data_dict, all_payment_features, 9)
totals_match(data_dict, all_stock_features, 3)
## There are 2 people in the dataset that have values that don't add up for payments or stock values

## correcting data

data_dict["BELFER ROBERT"]["salary"] = 'NaN'
data_dict["BELFER ROBERT"]["bonus"] = 'NaN'
data_dict["BELFER ROBERT"]["long_term_incentive"] = 'NaN'
data_dict["BELFER ROBERT"]["deferred_income"] = -102500
data_dict["BELFER ROBERT"]["deferral_payments"] = 'NaN'
data_dict["BELFER ROBERT"]["loan_advances"] = 'NaN'
data_dict["BELFER ROBERT"]["other"] = 'NaN'
data_dict["BELFER ROBERT"]["expenses"] = 3285
data_dict["BELFER ROBERT"]["director_fees"] = 102500
data_dict["BELFER ROBERT"]["total_payments"] = 3285
data_dict["BELFER ROBERT"]["exercised_stock_options"] = 'NaN'
data_dict["BELFER ROBERT"]["restricted_stock"] = 44093
data_dict["BELFER ROBERT"]["restricted_stock_deferred"] = -44093
data_dict["BELFER ROBERT"]["total_stock_value"] = 'NaN'




data_dict["BHATNAGAR SANJAY"]["salary"] = 'NaN'
data_dict["BHATNAGAR SANJAY"]["bonus"] = 'NaN'
data_dict["BHATNAGAR SANJAY"]["long_term_incentive"] = 'NaN'
data_dict["BHATNAGAR SANJAY"]["deferred_income"] = 'NaN'
data_dict["BHATNAGAR SANJAY"]["deferral_payments"] = 'NaN'
data_dict["BHATNAGAR SANJAY"]["loan_advances"] = 'NaN'
data_dict["BHATNAGAR SANJAY"]["other"] = 'NaN'
data_dict["BHATNAGAR SANJAY"]["expenses"] = 137864
data_dict["BHATNAGAR SANJAY"]["director_fees"] = 'NaN'
data_dict["BHATNAGAR SANJAY"]["total_payments"] = 137864
data_dict["BHATNAGAR SANJAY"]["exercised_stock_options"] = 15456290
data_dict["BHATNAGAR SANJAY"]["restricted_stock"] = 2604490
data_dict["BHATNAGAR SANJAY"]["restricted_stock_deferred"] = -2604490
data_dict["BHATNAGAR SANJAY"]["total_stock_value"] = 15456290


## Now lets test the totals again 
totals_match(data_dict, all_payment_features, 9)
totals_match(data_dict, all_stock_features, 3)

#No bad entries show now


##outlier examination
## salary,bonus
outlier_examination('salary', 'bonus', data_dict)
##4 outliers 2 are poi and 2 are not.
## the two that are not poi are:
## LAVORATO JOHN J has an unusually high bonus and FREVERT MARK A has an unusually high salary for Non POI's    

## salary,exercised_stock_options
outlier_examination('salary', 'exercised_stock_options', data_dict)
## 4 outliers, only 1 not a poi:
## FREVERT MARK A has an unusually high salary for Non POI's

## salary,restricted_stock
outlier_examination('salary', 'restricted_stock', data_dict)
## 5 outliers of which 3 are non poi
## WHITE JR THOMAS E and PAI LOU L have an unusually high amount of restricted stock
## FREVERT MARK A has an unusually high salary for Non POI's

## total_stock_value,total_payments
outlier_examination('total_stock_value', 'total_payments', data_dict)
## Using the above to check totals and it seems like all data points are valid
## 1 big outlier, LAY KENNETH L which is a poi
## Zooming in to exclude LAY and there are 6 other outliers of which  4 are not poi's:
## FREVERT MARK A, LAVORATO JOHN J AND MARTIN AMANDA K all have high total payments
## FREVERT MARK high total payments and high total stock,
## PAI LOU L has high total stocks.

## uses the percentage_comms_poi function found in intermediate_code.py 
percentage_comms_poi(data_dict)


poi_freq_sanity_check(data_dict, True, 0.1)
poi_freq_sanity_check(data_dict, False, 0.1)

## almost 78% of poi's have communication frequencies TO other poi's with only about 6% FROM
## while only 23% of non poi's have communication TO poi's and around 5% FROM
## let's increase frequency and see how the trend holds up

poi_freq_sanity_check(data_dict, True, 0.2)
poi_freq_sanity_check(data_dict, False, 0.2)

## almost 67% of poi's have this higher frequency of communication TO, while FROM drops to 0
## nonpoi's show a similar drop to just over 18% to and less than 1% from poi's
## Let's increase frequency further to see how the trend continues

poi_freq_sanity_check(data_dict, True, 0.3)
poi_freq_sanity_check(data_dict, False, 0.3)

## Here we see a big drop for both
## poi TO poi communication drops to just over 33% with less than 10% nonpoi TO poi
## the FROM rate drops to 0 for both


poi_freq_sanity_check(data_dict, True, 0.6)
poi_freq_sanity_check(data_dict, False, 0.6)
## When looking at a threshhold of 60% the comms ratios drops to just over 5% to poi's from poi's and just over 2 percent from non poi's to poi's. With 0% from
## I am not convinced this feature has much determinant value especially given almost a third of poi's lack email meta data, but if they do percent_to_poi will
## probably be more valuable than percent_from_poi


test_flist = ['poi','salary', 'to_messages', 'total_payments','deferral_payments',
              'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',
              'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances',
              'from_messages', 'other', 'from_this_person_to_poi','director_fees',
              'deferred_income','long_term_incentive','from_poi_to_this_person', 'percent_to_poi', 'percent_from_poi']

from sklearn.feature_selection import f_classif
## new dictionary with scaled data
temp_scaled_data = feature_scaling(data_dict, test_flist_ordered, 21)
### Store to temp_dataset
temp_dataset = temp_scaled_data
### Extract features and labels from dataset for local testing
data = featureFormat(temp_dataset, test_flist_ordered, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
grid = SelectKBest(f_classif).fit(features_train, labels_train)
print grid.scores_


# [  1.58583956e+01   2.61613809e+00   8.96141166e+00   1.10210083e-02
#    9.68026043e+00   3.07287974e+01   8.05496353e+00   1.07226847e+01
#    1.94304716e+00   1.06327580e+01   4.17386908e+00   7.03793093e+00
#    4.35307622e-01   3.20455696e+00   1.11182142e-01   1.81808008e+00
#    1.48018515e+00   7.55509239e+00   4.95872548e+00   1.58380708e+01
#    5.19324793e-01]


test_flist2 = ['poi','salary',  'bonus', 'shared_receipt_with_poi',  'total_stock_value', 'percent_to_poi' ]
## Lets test this with a classifier

## new dictionary with scaled data
temp_scaled_data = feature_scaling(data_dict, test_flist2, 4)
## Store to temp_dataset
temp_dataset = temp_scaled_data
## Extract features and labels from dataset for local testing
data = featureFormat(temp_dataset, test_flist2, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
gnbTest_k = GaussianNB()
GaussNB = classify(temp_dataset, test_flist2, features_train, labels_train, features_test, labels_test, clf=gnbTest_k)
## our metrics of interest are precision and recall
##  Precision: 0.53739  Recall: 0.25150

## We get a high precision score, but recall is too low

## Lets check outliers for the features we have not tested for yet: shared_receipt_with_poi and percent_to_poi

outlier_examination('shared_receipt_with_poi', 'percent_to_poi', data_dict)

## removing addiotnal outliers

## LAVORATO JOHN J unusually high salary, bonus and shared_receipt_with_poi for a non_poi
data_dict.pop("LAVORATO JOHN J", 0)

## FREVERT MARK A unusually high salary and total_stock_value
data_dict.pop("FREVERT MARK A", 0)

## PAI LOU L high total_stock_value
data_dict.pop("PAI LOU L", 0)

## HUMPHREY GENE E has an extremely high percent_to_poi
data_dict.pop("HUMPHREY GENE E", 0)

## KITCHEN LOUISE, KEAN STEVEN J, WHALLEY LAWERENCE G and SHAPIRO RICHARD S have high shared_receipt_with_poi
data_dict.pop("KITCHEN LOUISE", 0)
data_dict.pop("KEAN STEVEN J", 0)
data_dict.pop("WHALLEY LAWERENCE G", 0)
data_dict.pop("SHAPIRO RICHARD S", 0)


## Re run classifier with additional outliers removed 
## new dictionary with scaled data
temp_scaled_data = feature_scaling(data_dict, test_flist2, 4)
## Store to temp_dataset
temp_dataset = temp_scaled_data
## Extract features and labels from dataset for local testing
data = featureFormat(temp_dataset, test_flist2, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
gnbTest_k = GaussianNB()
GaussNB = classify(temp_dataset, test_flist2, features_train, labels_train, features_test, labels_test, clf=gnbTest_k)
## our metrics of interest are precision and recall
## Precision: 0.63402  Recall: 0.29450
## Precision and recall both increased significantly


# Let's look at the numbers of people in the dataset now:
# get the length of each list and total dataset
print len(people_lists(data_dict)[0])
# 18
print len(people_lists(data_dict)[1])
#118
print len(data_dict)
#136
## We have 136 people in the datset in total with 18 (13.23%) poi's and 118 (86.76%) non-poi's. 


# # ##Let's try with some additional features with high scores: specifically adding in exercised_stock_options, total_payments and restricted_stock alone and together
test_flist2 = ['poi','salary',  'bonus', 'shared_receipt_with_poi',  'total_stock_value', 'percent_to_poi' ]
test_flist3 = ['poi','salary',  'bonus', 'exercised_stock_options', 'shared_receipt_with_poi', 'total_stock_value', 'percent_to_poi' ]
test_flist4 = ['poi','salary',  'bonus', 'exercised_stock_options', 'total_payments', 'shared_receipt_with_poi',  'total_stock_value', 'percent_to_poi' ]
test_flist5 = ['poi','salary',  'bonus', 'exercised_stock_options', 'total_payments', 'restricted_stock', 'shared_receipt_with_poi',  'total_stock_value', 'percent_to_poi' ]
test_flist6 = ['poi','salary',  'bonus', 'total_payments', 'shared_receipt_with_poi', 'total_stock_value', 'percent_to_poi' ]
test_flist7 = ['poi','salary',  'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'total_stock_value', 'percent_to_poi' ]
test_flist8 = ['poi','salary',  'bonus', 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi', 'total_stock_value', 'percent_to_poi' ]

## code below commented out as this code will remove some outliers that I may not want removed in the final dataset

# ## Re run classifier with exercised_stock_options feature added 
# ## new dictionary with scaled data
# temp_scaled_data = feature_scaling(data_dict, test_flist3, 6)
# ## Store to temp_dataset
# temp_dataset = temp_scaled_data
# ## Extract features and labels from dataset for local testing
# data = featureFormat(temp_dataset, test_flist3, sort_keys = True)
# labels, features = targetFeatureSplit(data)

# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
# gnbTest_k = GaussianNB()
# GaussNB = classify(temp_dataset, test_flist3, features_train, labels_train, features_test, labels_test, clf=gnbTest_k)
# ## our metrics of interest are precision and recall
# ## Precision: 0.52012   Recall: 0.44600
# ## Precision and recall increased significantly 



# ## Re run classifier with total_payments feature added 
# ## new dictionary with scaled data
# temp_scaled_data = feature_scaling(data_dict, test_flist4, 7)
# ## Store to temp_dataset
# temp_dataset = temp_scaled_data
# ## Extract features and labels from dataset for local testing
# data = featureFormat(temp_dataset, test_flist4, sort_keys = True)
# labels, features = targetFeatureSplit(data)

# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
# gnbTest_k = GaussianNB()
# GaussNB = classify(temp_dataset, test_flist4, features_train, labels_train, features_test, labels_test, clf=gnbTest_k)
## our metrics of interest are precision and recall
## Precision: 0.41687   Recall: 0.34850
## Precision and recall drop with total_payments added

# ## Let's remve outlier's associated with the extra features
# ## MARTIN AMANDA K unusually high total payments
# data_dict.pop("MARTIN AMANDA K", 0)

# ## Re run classifier with total_payments feature added and associated outliers removed
# ## new dictionary with scaled data
# temp_scaled_data = feature_scaling(data_dict, test_flist4, 7)
# ## Store to temp_dataset
# temp_dataset = temp_scaled_data
# ## Extract features and labels from dataset for local testing
# data = featureFormat(temp_dataset, test_flist4, sort_keys = True)
# labels, features = targetFeatureSplit(data)

# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
# gnbTest_k = GaussianNB()
# GaussNB = classify(temp_dataset, test_flist4, features_train, labels_train, features_test, labels_test, clf=gnbTest_k)
# ## our metrics of interest are precision and recall
## Precision: 0.43048  Recall: 0.35450
## Precision and recall increase slightly but still not as good as without total_payments


## Re run classifier with additional restricted_stock feature added 
# ## Let's remve outlier's associated with the extra features
# ## WHITE JR THOMAS E unusually high restricted stock
# data_dict.pop("WHITE JR THOMAS E", 0)
# ## new dictionary with scaled data
# temp_scaled_data = feature_scaling(data_dict, test_flist5, 8)
# ## Store to temp_dataset
# temp_dataset = temp_scaled_data
# ## Extract features and labels from dataset for local testing
# data = featureFormat(temp_dataset, test_flist5, sort_keys = True)
# labels, features = targetFeatureSplit(data)

# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
# gnbTest_k = GaussianNB()
# GaussNB = classify(temp_dataset, test_flist5, features_train, labels_train, features_test, labels_test, clf=gnbTest_k)
# ## our metrics of interest are precision and recall
# ## Precision: 0.46037 Recall: 0.39500
# ## Precision and recall increased but still not as good as without total_payments or restricted_stock

# # Re run classifier with only additional total_payments feature added 
# ## Let's remve outlier's associated with the extra features
# ## MARTIN AMANDA K unusually high total payments
# data_dict.pop("MARTIN AMANDA K", 0)
# ## new dictionary with scaled data
# temp_scaled_data = feature_scaling(data_dict, test_flist6, 6)
# ## Store to temp_dataset
# temp_dataset = temp_scaled_data
# ## Extract features and labels from dataset for local testing
# data = featureFormat(temp_dataset, test_flist6, sort_keys = True)
# labels, features = targetFeatureSplit(data)
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
# gnbTest_k = GaussianNB()
# GaussNB = classify(temp_dataset, test_flist6, features_train, labels_train, features_test, labels_test, clf=gnbTest_k)
# ## our metrics of interest are precision and recall
# ## Precision: 0.44058   Recall: 0.34850
# ## Precision and recall not as good as with exercised_stock_options


# # Re run classifier with only additional restricted_stock feature added 
# ## Let's remve outlier's associated with the extra features
# ## WHITE JR THOMAS E unusually high restricted stock
# data_dict.pop("WHITE JR THOMAS E", 0)
# ## new dictionary with scaled data
# temp_scaled_data = feature_scaling(data_dict, test_flist7, 7)
# ## Store to temp_dataset
# temp_dataset = temp_scaled_data
# ## Extract features and labels from dataset for local testing
# data = featureFormat(temp_dataset, test_flist7, sort_keys = True)
# labels, features = targetFeatureSplit(data)
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
# gnbTest_k = GaussianNB()
# GaussNB = classify(temp_dataset, test_flist7, features_train, labels_train, features_test, labels_test, clf=gnbTest_k)
# ## our metrics of interest are precision and recall
# ## Precision: 0.49558   Recall: 0.39250
# ## Precision and recall increased vs total payments but not as good as with exercised_stock_options


# # Re run classifier with only additional exercised_stock_options and restricted_stock feature added 
# ## Let's remve outlier's associated with the extra features
# ## WHITE JR THOMAS E unusually high restricted stock
# data_dict.pop("WHITE JR THOMAS E", 0)
# ## new dictionary with scaled data
# temp_scaled_data = feature_scaling(data_dict, test_flist8, 8)
# ## Store to temp_dataset
# temp_dataset = temp_scaled_data
# ## Extract features and labels from dataset for local testing
# data = featureFormat(temp_dataset, test_flist8, sort_keys = True)
# labels, features = targetFeatureSplit(data)
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
# gnbTest_k = GaussianNB()
# GaussNB = classify(temp_dataset, test_flist8, features_train, labels_train, features_test, labels_test, clf=gnbTest_k)
# ## our metrics of interest are precision and recall
# ## Precision: 0.48840   Recall: 0.38950
# ## Precision and recall decreased vs restricted stock alone and not as good as with exercised_stock_options


## so it looks like exercised_stock_options is worth adding, but restricted_stock and total_payments are not.
## Our final feature set is

features_list = ['poi','salary',  'bonus', 'exercised_stock_options', 'shared_receipt_with_poi', 'total_stock_value', 'percent_to_poi' ]

#Now lets run the peoples list code again to see how many values we now have:
# get the length of each list and total dataset
print len(people_lists(data_dict)[0])
# 18
print len(people_lists(data_dict)[1])
#118
print len(data_dict)
136

## so it looks like exercised_stock_options is worth adding, but restricted_stock and total_payments are not.
## Our final feature set is

features_list = ['poi','salary',  'bonus', 'exercised_stock_options', 'shared_receipt_with_poi', 'total_stock_value', 'percent_to_poi' ]

##Now lets run the peoples list code again to see how many values we now have:
## We now have 18 POI's and 118 non-POI's out of a total dataset of 136 people with 6 features and 1 label (poi) 

## new dictionary with scaled data
scaled_data = feature_scaling(data_dict, features_list, 6)


### Store to my_dataset for easy export below.
my_dataset = scaled_data           
            

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

##### Task 4: Try a varity of classifiers
##### Please name your classifier clf for easy export below.



features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


##1st classifier GaussianNB
##Note has no tunable parameters except priors which I will investigate later as GaussianNB has
##few options for tuning and it already hits the > 0.3 target

gnb = GaussianNB()
GaussNB = classify(my_dataset, features_list, features_train, labels_train, features_test, labels_test, clf=gnb)

## Results:
## Accuracy: 0.85146
## Precision: 0.52012    Recall: 0.44600
## F1: 0.48022   F2: 0.45908
## Total predictions: 13000
## True positives:  892  False positives:  823
## False negatives: 1108 True negatives: 10177
## best score:  0.855855855856
## Classification Report:
## classes  precision    recall  f1-score   support
##
##  0.0       0.94      0.91      0.93        35
##  1.0       0.25      0.33      0.29         3
##
## avg/total  0.89      0.87      0.88        38


## 2nd classifier test: AdaBoost with Decision tree as base estimator
params= {'n_estimators':[2,3,4,5,6, 10, 15] }
abd = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', min_samples_split=4, max_depth=2),algorithm = "SAMME", random_state = 42)
AdaBoost = classify(my_dataset, features_list, features_train, labels_train, features_test, labels_test, clf=abd, params=params)
## Results:
## Accuracy: 0.83008
## Precision: 0.43481   Recall: 0.34850
## F1: 0.38690  F2: 0.36291
## Total predictions: 13000
## True positives:  697 False positives:  906
## False negatives: 1303    True negatives: 10094
## best score:  0.828828828829
## best params: {'n_estimators': 3}
## Classification Report:
## classes  precision    recall  f1-score   support
##
##  0.0       0.97      0.91      0.94        35
##  1.0       0.40      0.67      0.50         3
##
## avg/total  0.92      0.89      0.91        38



## 3rd Classifier Random Forest
params={'n_estimators':[2,3,4,5,6, 10, 15],
       'min_samples_leaf':[2,3,4],
       'max_leaf_nodes':[None, 2, 4, 6]}
rfst = RandomForestClassifier(bootstrap=True, criterion = 'entropy', max_features=None)
RForest = classify(my_dataset, features_list, features_train, labels_train, features_test, labels_test, clf=rfst, params = params)
## Results:
## Accuracy: 0.84008
## Precision: 0.46176    Recall: 0.23850
## F1: 0.31454   F2: 0.26403
## Total predictions: 13000
## True positives:  477  False positives:  556
## False negatives: 1523 True negatives: 10444
## best score:  0.873873873874
## best params: {'n_estimators': 5, 'max_leaf_nodes': None, 'min_samples_leaf': 4}
## Classification Report:
## classes   precision    recall  f1-score   support
##  0.0       0.94      0.97      0.96        35
##  1.0       0.50      0.33      0.40         3
## avg/total  0.91      0.92      0.91        38


## 4th classifier test: AdaBoost with Decision tree as base estimator, recall focus
params= {'n_estimators':[2,3,4,5,6, 10, 15] }
abdcr = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', min_samples_split=4, max_depth=2),algorithm = "SAMME", random_state = 42)
AdaBoost = classify(my_dataset, features_list, features_train, labels_train, features_test, labels_test, clf=abdcr, params=params, score_method='recall')
## Results:
## Accuracy: 0.82531
## Precision: 0.41320   Recall: 0.32250
## F1: 0.36226  F2: 0.33731
## Total predictions: 13000
## True positives:  645 False positives:  916
## False negatives: 1355    True negatives: 10084
## best score:  0.316516516517
## best params: {'n_estimators': 3}
## Classification Report:
## classes  precision    recall  f1-score   support
##  0.0       0.97      0.91      0.94        35
##  1.0       0.40      0.67      0.50         3
## avg/total  0.92      0.89      0.91        38

##5th classifier Random forest with recall focus
params={'n_estimators':[2,3,4,5,6, 10, 15],
       'min_samples_leaf':[2,3,4],
       'max_leaf_nodes':[None, 2, 4, 6]}
rfstr = RandomForestClassifier(bootstrap=True, criterion = 'entropy', max_features=None)
RForest = classify(my_dataset, features_list, features_train, labels_train, features_test, labels_test, clf=rfstr, params = params, score_method='recall')
## Results
## Accuracy: 0.83346
## Precision: 0.43108    Recall: 0.25800
## F1: 0.32280  F2: 0.28053
## Total predictions: 13000
## True positives:  516 False positives:  681
## False positives:  681    False negatives: 1484   True negatives: 10319
## best score:  0.384984984985
## best params: {'n_estimators': 4, 'max_leaf_nodes': 6, 'min_samples_leaf': 4}
##Classification Report:
## classes   precision    recall  f1-score   support
##  0.0        0.94      0.97      0.96        35
##  1.0        0.50      0.33      0.40         3
##avg/total    0.91      0.92      0.91        38


### Task 5: Tune classifier to achieve better than .3 precision and recall 
### using testing script tester.py .

##1st tuning attempt: GaussianNB  parameter: prior

gnb2 = GaussianNB()
params = {'priors':[[0.1,0.9],[0.2,0.8],[0.3,0.7],[0.4,0.6],[0.5,0.5],[0.6,0.4],[0.7,0.3],[0.8,0.2],[0.9,0.1],[0.1324,0.8676],[0.8676,0.1324]]}
GaussNB = classify(my_dataset, features_list, features_train, labels_train, features_test, labels_test, clf=gnb2, params=params)

## Results:
## Accuracy: 0.84054
## Precision: 0.48041    Recall: 0.44750
## F1: 0.46337  F2: 0.45372
## Total predictions: 13000
## True positives:  895 False positives:  968
## False negatives: 1105    True negatives: 10032

## best score:  0.864864864865
## best params: {'priors': [0.8, 0.2]}
## Classification Report:
## classes  precision    recall  f1-score   support

##  0.0       0.94      0.91      0.93        35
##  1.0       0.25      0.33      0.29         3
##
## avg/total  0.89      0.87      0.88        38
## slight improvement in recall but big loss in precision




## 2nd tuning attempt this time on GaussianNB scoring using recall

gnb3 = GaussianNB()
GaussNB = classify(my_dataset, features_list, features_train, labels_train, features_test, labels_test, clf=gnb3, score_method='recall')

## Results:
## Accuracy: 0.85146
## Precision: 0.52012   Recall: 0.44600
## F1: 0.48022  F2: 0.45908
## Total predictions: 13000
## True positives:  892 False positives:  823
## False negatives: 1108    True negatives: 10177
## best score:  0.375375375375
## scorer: make_scorer(recall_score)
## Classification Report:
## classes  precision    recall  f1-score   support
##   0.0       0.94      0.91      0.93        35
##   1.0       0.25      0.33      0.29         3
## avg/total   0.89      0.87      0.88        38


## 3rd tuning attempt this time on GaussianNB " parameter: prior, plus gridsearchCV parameter scoring using recall

gnb4 = GaussianNB()
params = {'priors':[[0.1,0.9],[0.2,0.8],[0.3,0.7],[0.4,0.6],[0.5,0.5],[0.6,0.4],[0.7,0.3],[0.8,0.2],[0.9,0.1],[0.1324,0.8676],[0.8676,0.1324]]}
GaussNB = classify(my_dataset, features_list, features_train, labels_train, features_test, labels_test, clf=gnb4, params=params, score_method='recall')
## Results:
## Accuracy: Accuracy: 0.78954
## Precision: 0.37742   Recall: 0.56650
## F1: 0.45302  F2: 0.51491
## Total predictions: 13000
## True positives: 1133 False positives: 1869
## False negatives:  867    True negatives: 9131
## best score:  0.546546546547
## best params: {'priors': [0.1, 0.9]}
## scorer: make_scorer(recall_score)
## Classification Report:
## classes  precision    recall  f1-score   support
##  0.0       0.94      0.91      0.93        35
##  1.0       0.25      0.33      0.29         3
## avg/total  0.89      0.87      0.88        38