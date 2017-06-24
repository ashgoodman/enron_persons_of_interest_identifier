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
from sklearn.ensemble import RandomForestClassifier 



### Task 1: Select features to use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


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


## using nanlist function to determine list of people who do not have a given feature
## code found in intermediate_code.py and executed with results shown in intermediate_steps.py 


## From this we can see that some variables have zero correlation to POI's
## These include restricted_stock_deferred and director_fees, they also have a very low correlation with non poi's
## So it makes sense to remove these two features
temp_feature_list_2 = ['poi','salary', 'total_payments','to_messages','deferral_payments',
                 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',
                 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 
                 'from_this_person_to_poi','deferred_income','long_term_incentive','from_poi_to_this_person'] 

## We can also see that some variables occur very infrequently with POI's
## These include deferral_payments and loan_advances, which occur 28% and 6% of the time respectively
## These variables also occur equally infrequently with non poi's 27% and 2 % respectively
## So it makes sense to remove these 2 features as well
temp_feature_list_3 = ['poi','salary', 'total_payments','to_messages', 'exercised_stock_options', 'bonus', 
                       'restricted_stock', 'shared_receipt_with_poi', 'total_stock_value', 'expenses', 'from_messages', 
                       'other', 'from_this_person_to_poi', 'deferred_income','long_term_incentive','from_poi_to_this_person']





        
### Task 2: Remove outliers

## From previous exploration during the ML coursework and looking at the pdf file "enron61702insiderpay.pdf"
## I know there are 2 entries that should be removed before looking for outliers. They are "TOTAL" and
## 'THE TRAVEL AGENCY IN THE PARK'

## TOTAL is the , as the name suggests, the totals for all features or columns

## THE TRAVEL AGENCY IN THE PARK was a company half owned by Ken Lay's sister and Enron was their largest client
## accounting for 50-80% of their annual revenue. Enron and Enron employees purchased over $100 million USD
## a year in travel through the agency.

## It is possible that this was used to launder some of the money from Enron, but that question can not be
## answered easily through the data available to me.


## Removing TOTAL and THE TRAVEL AGENCY IN THE PARK
data_dict.pop( "TOTAL", 0 )
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

## using people_lists found in intermediate_code.py and executed in intermediate_steps.py
## We now have 18 POI's and 126 non-POI's out of a total dataset of 144 people with 15 features and 1 label (poi) 

## Next lets see how many people have empty values for all features
## Looking for anyone missing all financial data using the all_nan function 
## code found in intermediate_code.py and executed with results shown in intermediate_steps.py 

## LOCKHART EUGENE E  is missing all financial data and email data and is not a poi, so I will remove him as an outlier
data_dict.pop("LOCKHART EUGENE E", 0)
## KOPPER MICHAEL J, FASTOW ANDREW S, YEAGER F SCOTT and HIRKO JOSEPH are all poi's who have no email meta data. As they are poi's I will retain them, 
## but this brings into question the usefullness of email meta data when almost 28% of poi's have no email meta data associated with them

## Data examination: are all financial values correct?
## to test this all payment info must add up to total payments value and all stock info must add up to total stock values
## uses totals_match function found in intermediate_code.py and executed with results shown in intermediate_steps.py


## There are 2 people in the dataset that have values that don't add up for payments or stock values

##payments:
#BELFER ROBERT 1642 102500
#BHATNAGAR SANJAY 7866009 15456290

##Stocks
#BELFER ROBERT 1642 -44093
#BHATNAGAR SANJAY 7728145 0

# ##Can these datapoints be corrected? I will compare what's in the dataset to whats in the pdf file "enron61702insiderpay.pdf"

#print data_dict["BELFER ROBERT"] 
# {'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': -102500, 'total_payments': 102500, 
# 'exercised_stock_options': 3285, 'bonus': 'NaN', 'restricted_stock': 'NaN', 'shared_receipt_with_poi': 'NaN', 
# 'restricted_stock_deferred': 44093, 'total_stock_value': -44093, 'expenses': 'NaN', 'loan_advances': 'NaN', 
# 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'director_fees': 3285, 
# 'deferred_income': 'NaN', 'long_term_incentive': 'NaN', 'email_address': 'NaN', 'from_poi_to_this_person': 'NaN'}
## BELFER has incorrect values for deferral_payments,deferred_income, expenses, director fees, restricted stock, exercised_stock_options, total_stock_value and total_payments
## Correcting to values found in "enron61702insiderpay.pdf"
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



#print data_dict["BHATNAGAR SANJAY"] 
# {'salary': 'NaN', 'to_messages': 523, 'deferral_payments': 'NaN', 'total_payments': 15456290, 
# 'exercised_stock_options': 2604490, 'bonus': 'NaN', 'restricted_stock': -2604490, 'shared_receipt_with_poi': 463, 
# 'restricted_stock_deferred': 15456290, 'total_stock_value': 'NaN', 'expenses': 'NaN', 'loan_advances': 'NaN', 
# 'from_messages': 29, 'other': 137864, 'from_this_person_to_poi': 1, 'poi': False, 'director_fees': 137864, 
# 'deferred_income': 'NaN', 'long_term_incentive': 'NaN', 'email_address': 'sanjay.bhatnagar@enron.com', 
# 'from_poi_to_this_person': 0}

## has incorrect values for expenses, director_fees, total_payments, exercised_stock_options, restricted_stock, restricted_stock_deferred and total_stock_value
## Correcting to values found in "enron61702insiderpay.pdf"
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


## Outlier examination
## uses the outlier_examination function found in intermediate_code.py and executed with results shown in intermediate_steps.py


## non poi outliers    
## LAVORATO JOHN J has an unusually high bonus and FREVERT MARK A has an unusually high salary for Non POI's    
## FREVERT MARK A has an unusually high salary for Non POI's
## WHITE JR THOMAS E and PAI LOU L have an unusually high amount of restricted stock
## FREVERT MARK A has an unusually high salary for Non POI's
## FREVERT MARK A, LAVORATO JOHN J AND MARTIN AMANDA K all have high total payments
## FREVERT MARK high total payments and high total stock,
## PAI LOU L has high total stocks.


## The question is: what to do with the outliers?
## On the one hand, removing them would give more determinant power for the associated features
## On the other hand, if we were trying to identify poi's without labels we wouldn't know these are non poi outliers...

## For now I will keep them as I investigate the data further. I may later decide to remove them


        
### Task 3: Create new feature(s)

## the ratio of messages a person sends vs the messages they sent to a POI
## and the ratio of the messages they recieved vs recieved from poi might be
## interesting to have, and more informative then just the raw numbers of sent and recieved
## from poi.


## uses the percentage_comms_poi function found in intermediate_code.py Note:peek-ahead can be an issue with these features as noted in the project writeup
## file enron_writeup.pdf
percentage_comms_poi(data_dict)


## As a sanity check I want to see how many poi's and non poi's have high frequencies of communications

## uses the poi_freq_sanity_check function found in intermediate_code.py and executed with results shown in intermediate_steps.py

## When looking at a threshhold of 60% the comms ratios drops to just over 5% to poi's from poi's and just over 2 percent from non poi's to poi's. With 0% from
## I am not convinced this feature has much determinant value especially given almost a third of poi's lack email meta data, but if they do percent_to_poi will
## probably be more valuable than percent_from_poi


##Let's test these new features and the features I have not yet eliminated along with the other features I have already removed using SelectKBest

## found in intermediate_steps.py with results below:
## ** = high importance. * = medium importance

## salary = 1.58583956e+01 **
## to_messages = 2.61613809e+00
## total_payments = 8.96141166e+00 *
## deferral_payments = 1.10210083e-02
## exercised_stock_options = 9.68026043e+00 *
## bonus = 3.07287974e+01 **
## restricted_stock = 8.05496353e+00 *
## shared_receipt_with_poi = 1.07226847e+01 **
## restricted_stock_deferred = 1.94304716e+00
## total_stock_value = 1.06327580e+01 **
## expenses = 4.17386908e+00
## loan_advances = 7.03793093e+00
## from_messages = 4.35307622e-01
## other = 3.20455696e+00
## from_this_person_to_poi = 1.11182142e-01
## director_fees = 1.81808008e+00
## deferred_income = 1.48018515e+00
## long_term_incentive = 7.55509239e+00
## from_poi_to_this_person = 4.95872548e+00
## percent_to_poi = 1.58380708e+01 **
## percent_from_poi = 5.19324793e-01


## Using SelectKBest we can see salary, bonus, shared_receipt_with_poi, total_stock_value and percent_to_poi have the highest k values
## This refutes my theory that email meta data may be of little value as 2/5th's of the features are email related, Note:peek-ahead can be an issue with 
## the use of the email features as noted in the project writeup file enron_writeup.pdf

test_flist2 = ['poi','salary',  'bonus', 'shared_receipt_with_poi',  'total_stock_value', 'percent_to_poi' ]
## Lets test this with a classifier (test found in intermediate_steps.py )

## test conducted with GaussianNB 
# ## our metrics of interest are precision and recall
# ##  Precision: 0.53739  Recall: 0.25150

## Lets check outliers for the features we have not tested for yet: shared_receipt_with_poi and percent_to_poi
## uses the outlier_examination function found in intermediate_code.py and executed with results shown in intermediate_steps.py

##The following non poi's have high values
## HUMPHREY GENE E has an extremely high percent_to_poi
## LAVORATO JOHN J, KITCHEN LOUISE, KEAN STEVEN J, WHALLEY LAWERENCE G and SHAPIRO RICHARD S have high shared_receipt_with_poi

## what if we remove the non-poi outliers identified earlier but not yet removed along with the new outliers?

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



## Note regarding outlier removal: peek-ahead due to foreknowledge of poi values for outliers 
## can be an issue with these features as noted in the project writeup file enron_writeup.pdf


## Re run classifier with additional outliers removed (test found in intermediate_steps.py )
## test conducted with GaussianNB 
## our metrics of interest are precision and recall
## Precision: 0.63402  Recall: 0.29450
## Precision and recall both increased significantly


## Let's look at the numbers of people in the dataset now:
## using people_lists found in intermediate_code.py and executed in intermediate_steps.py
## We have 136 people in the datset in total with 18 (13.23%) poi's and 118 (86.76%) non-poi's. 

## Re run GaussianNB classifier using various additonal features as shown in intermediate_steps.py  results as follows:

## with exercised_stock_options feature added
## Precision: 0.52012   Recall: 0.44600
## Precision and recall increased significantly 

## with exercised_stock_options and total_payments feature added and total_payments associated outliers removed
## Precision: 0.43048  Recall: 0.35450
## Precision and recall increase slightly but still not as good as without total_payments

## with additional restricted_stock feature added in addition to exercised_stock_options and total_payments and associated outliers for restricted_stock removed
## Precision: 0.46037 Recall: 0.39500
## Precision and recall increased but still not as good as without total_payments or restricted_stock

## with only additional total_payments feature added and associated outliers removed
## Precision: 0.44058   Recall: 0.34850
## Precision and recall not as good as with exercised_stock_options

## with only additional restricted_stock feature added and associated outliers removed
## Precision: 0.49558   Recall: 0.39250
## Precision and recall increased vs total payments but not as good as with exercised_stock_options


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


## Testing various classifiers shown in intermediate_steps.py
## results of each shown below:

##1st classifier GaussianNB
## Precision: 0.52012    Recall: 0.44600


## 2nd classifier test: AdaBoost with Decision tree as base estimator
## Precision: 0.43481   Recall: 0.34850


## 3rd Classifier Random Forest
## Precision: 0.46176    Recall: 0.23850


## 4th classifier test: AdaBoost with Decision tree as base estimator, recall focus
## Precision: 0.41320   Recall: 0.32250


##5th classifier Random forest with recall focus
## Precision: 0.43108    Recall: 0.25800



## Looking at the results so far and several of the classifiers met or came close to meeting the 0.3 target for recall and precision.
## Of the 2 metrics (recall/precision) I need to decide wich one is the most important to optimize for.
## Precision = TP/(TP+FP)
## Recall = TP/(TP+FN)
## which is most important to pay attention to? False positives or false negatives?
## The answer is dependant on the problem type. If you are looking for guilt, or risk then it would be better to have a higher number of false postives than false negatives. 
## False negatives means you mis-classified a risk as not a risk, whether looking at credit card transactions for fraud, human behavior as with enron, or terrorism
## It is better to classify someone as potentially guilty (a poi) and manually look at them closer to verify if they were classfied correctly 
## than to classify them as innnocent and ignore them altogether.


## If we assume the following assertions are true
## 1) There will, generally speaking, be far more innocents than guilty in most situations. 
## 2) Human attention resources (human oversight capacity) are limited.
## 3) mis classifying a guilty person or fraudulant transaction as not guilty/fraudulant carries far more risk than mis classifying an innocent person/transation a guilty
## Therfore, in my opinion, it is better to emphasize recall in situations like this providing the nagative impact onprecision isn't to great

## Of the classifiers I have run so far the best results in the precision/recall tradeoff came from Naive Bayes.
## Precision: 0.52012   Recall: 0.44600
## It was also the most efficient/fastest in computational power

## I will focus on tuning Naive Bayes



### Task 5: Tune classifier to achieve better than .3 precision and recall 
### using testing script tester.py . 

## Tuning shown in intermediate_steps.py  results shown below

##1st tuning attempt: GaussianNB parameter: prior
## Precision: 0.48041    Recall: 0.44750
## slight improvement in recall but big loss in precision


## 2nd tuning attempt this time on GaussianNB, no parameters but with scoring using recall
## Precision: 0.52012   Recall: 0.44600
## NO real change from vanilla GaussianNB results

## 3rd tuning attempt this time on GaussianNB parameter: prior, plus gridsearchCV parameter scoring using recall
## Precision: 0.37742   Recall: 0.56650
## Big increase in recall with big decrease in precision, best params: {'priors': [0.1, 0.9]}

## Of the 3 tunings the recall scoring focus with priors[0.1, 0.9] gave the best results for recall while still maintaining fair precision
## Precision: 0.37742   Recall: 0.56650 and Accuracy: 0.78954



clf = GaussianNB( priors = [0.1, 0.9])
clf = GridSearchCV(clf, param_grid = {}, scoring='recall')
clf = clf.fit(features_train, labels_train)
test_classifier(clf, my_dataset, features_list)




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check results. Ensure generation of the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)