#####################################################################
# This code is adapted from the original code used in the Hyperaktiv
# experimemts. It imports functions from their utils script.
# This contains the code used to create a large number of plots in
# order to investigate data leakage. 
# It also has some especially written functions for analysing 
# correlations between certain variables
######################################################################



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tsfresh.feature_selection.selection import select_features


import re
from utils import k_fold_model_evaluation

def reduce_dimensions(dataX, thres):
    correlations = dataX.corr().abs()
    mask = np.triu(np.ones_like(correlations, dtype=bool))

    tri_df = correlations.mask(mask)
    correlations = correlations.reset_index()
    to_drop = [c for c in tri_df.columns if any(tri_df[c]>thres)]
    
    dataX = dataX.drop(to_drop, axis=1)
    return dataX

def clean_length_cols(dataX, thresh):
    to_drop = []
    per_cor = []
    for col in dataX.columns:
        cor = dataX[col].corr(dataX["ACC__length"])
        if cor != 1 and cor >=thresh:
            to_drop.append(col)
        if cor ==1:
            per_cor.append(col)
    dataX = dataX.drop(to_drop, axis=1)
    dataX = dataX.drop(per_cor, axis=1)
    return dataX


# setting random seed for random state to 0 
random_seed = 0

# setting testing percentage of data
test_ratio = 0.3

# Setting number of splits for k fold crossvalidataion
k_folds = 10

_PARAMS_LORGREG = {
   "penalty": "l2", "C": 1.0, "class_weight": "balanced",
   "random_state": 0, "solver": "liblinear", "n_jobs": 1
    }

_PARAMS_RFC = {
    "n_estimators": 1000,
    "max_features": None, "max_depth": None,
    "min_samples_split": 2, "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_leaf_nodes": None, "bootstrap": True,
    "oob_score": False, "n_jobs": -1, "random_state": 0,
    "class_weight": "balanced"
    }

_PARAMS_XGB = {
    "random_state": random_seed, "verbosity": 0,
    'objective':'binary:logistic'
    }

_PARAMS_LIGHTGB = {
    "random_state": random_seed, "verbosity": 0,
    "objective": "binary"
    }


data = pd.read_csv("D:\Data Science MTU\Final Project\hyperaktiv\Current\Provided data\orig_features.csv", index_col="ID")
data = data.drop("col1", axis=1)


data = data.rename(columns = lambda x:re.sub('"', '', x))
data = data.rename(columns = lambda x:re.sub(',', '', x))

dataX = data.drop("ADHD", axis=1)
dataX = dataX.fillna(0)
dataY = data["ADHD"]

#na_sum = dataX.isnull().sum(axis=0) # looking for columns with na values

#dataX.columns
# Dropping columns that have na values
# for column in dataX.columns:
#     if dataX[column].isnull().sum(axis=0) > 0:
#         dataX = dataX.drop([column], axis=1)

# na_sum = dataX.isnull().sum(axis=0)     
# Find relevant features using tsfresh

dataX = dataX.drop(['ACC__first_location_of_minimum', "ACC__augmented_dickey_fuller__attr_pvalue__autolag_AIC", ], axis=1)

#dataX = reduce_dimensions(dataX, 0.7)

dataX = select_features(dataX, dataY)
length = dataX["ACC__length"]
dataX = clean_length_cols(dataX,0.7)

dataX = dataX.join(length)
c = dataX.columns

dataX.plot(kind="scatter" , y="ACC__length", 
           x="ACC__augmented_dickey_fuller__attr_teststat__autolag_AIC", 
           title="Plot of Augmented Dickey Fuller Test Statistic versus Length of timeseries")


dataX.plot(kind="scatter" , y="ACC__length", 
           x="ACC__ratio_value_number_to_time_series_length", 
           title="Plot Ratio value Number versus Length of timeseries")

dataX.plot(kind="scatter" , y="ACC__length", 
           x="ACC__first_location_of_minimum", 
           title="Plot of First Location of Minimum versus Length of timeseries")

dataX = dataX.drop(59, axis=0)
x = dataX.columns
dataX.plot(kind="scatter" , y="ACC__length", 
           x="ACC__augmented_dickey_fuller__attr_pvalue__autolag_AIC", 
           title="Plot of Augmented Dickey Fuller p-value with autolag: AIC versus Length of timeseries")


for col in c:
    dataX.plot(kind="scatter" ,y="ACC__length", 
               x=col, 
               title=col)

#relevant = calculate_relevance_table(dataX, dataY)

#dimensions = pd.concat([dataX,dataY],axis=1)
#dimensions.to_csv("D:\Data Science MTU\Final Project\hyperaktiv\Dataset\dimensions.csv")

#dataX = dataX.drop("ACC__augmented_dickey_fuller__attr_pvalue__autolag_AIC", axis=1)


dataX.plot(x="ACC__augmented_dickey_fuller__attr_pvalue__autolag_AIC", y="ACC__augmented_dickey_fuller__attr_teststat__autolag_AIC", style='o')
#dataX.columns

data.plot(y='ACC__augmented_dickey_fuller__attr_teststat__autolag_AIC', x="ACC__length", style='o')
data.plot(y='ACC__fft_coefficient__attr_abs__coeff_97', x="ACC__length", style='o')

dataX = dataX[['ACC__augmented_dickey_fuller__attr_teststat__autolag_AIC','ACC__fft_coefficient__attr_abs__coeff_97']]    

scaler = StandardScaler(copy=True)
dataX.loc[:, dataX.columns] = scaler.fit_transform(dataX[dataX.columns])
    
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        dataX,
        dataY,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=dataY)
        
metric_names = ["ACC", "PREC", "REC", "F1", "MCC"]

stratified_train_eval, stratified_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "stratified", "random_state": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits= k_folds, random_state=random_seed)
most_frequent_train_eval, most_frequent_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "most_frequent", "random_state": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=k_folds, random_state=random_seed)
prior_train_eval, prior_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "prior", "random_state": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=k_folds, random_state=random_seed)
minor_train_eval, minor_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "constant", "random_state": 0, "constant": 1 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=k_folds, random_state=random_seed)
major_train_eval, major_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "constant", "random_state": 0, "constant": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=k_folds, random_state=random_seed)
random_train_eval, random_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "uniform", "random_state": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=k_folds, random_state=random_seed)

logreg_train_eval, logreg_test_eval = k_fold_model_evaluation(LogisticRegression, _PARAMS_LORGREG,
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=k_folds, random_state=random_seed)
rfc_train_eval, rfc_test_eval = k_fold_model_evaluation(RandomForestClassifier, _PARAMS_RFC,
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=k_folds, random_state=random_seed)
xgb_train_eval, xgb_test_eval = k_fold_model_evaluation(XGBClassifier, _PARAMS_XGB,
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=k_folds, random_state=random_seed)
gbm_train_eval, gbm_test_eval = k_fold_model_evaluation(LGBMClassifier, _PARAMS_LIGHTGB,
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=k_folds, random_state=random_seed)

print("***CROSS-VALIDATION PERFORMANCE***")
print("MODEL\t" + "\t ".join(metric_names))
print("Rand\t" + "\t ".join([ "%.2f" % (np.mean(random_train_eval[name])) for name in metric_names ]))
print("Strat\t" + "\t ".join([ "%.2f" % (np.mean(stratified_train_eval[name])) for name in metric_names ]))
print("Minor\t" + "\t ".join([ "%.2f" % (np.mean(minor_train_eval[name])) for name in metric_names ]))
print("Major\t" + "\t ".join([ "%.2f" % (np.mean(major_train_eval[name])) for name in metric_names ]))
print("Prior\t" + "\t ".join([ "%.2f" % (np.mean(prior_train_eval[name])) for name in metric_names ]))
print("LogReg\t" + "\t ".join([ "%.2f" % (np.mean(logreg_train_eval[name])) for name in metric_names ]))
print("RFC\t\t" + "\t ".join([ "%.2f" % (np.mean(rfc_train_eval[name])) for name in metric_names ]))
print("XGB\t\t" + "\t ".join([ "%.2f" % (np.mean(xgb_train_eval[name])) for name in metric_names ]))
print("GBM\t\t" + "\t ".join([ "%.2f" % (np.mean(gbm_train_eval[name])) for name in metric_names ]))

print("\n")

print("***TEST PERFORMANCE***")
print("MODEL\t" + "\t ".join(metric_names))
print("Rand\t" + "\t ".join([ "%.2f" % (np.mean(random_test_eval[name])) for name in metric_names ]))
print("Strat\t" + "\t ".join([ "%.2f" % (np.mean(stratified_test_eval[name])) for name in metric_names ]))
print("Minor\t" + "\t ".join([ "%.2f" % (np.mean(minor_test_eval[name])) for name in metric_names ]))
print("Major\t" + "\t ".join([ "%.2f" % (np.mean(major_test_eval[name])) for name in metric_names ]))
print("Prior\t" + "\t ".join([ "%.2f" % (np.mean(prior_test_eval[name])) for name in metric_names ]))
print("LogReg\t" + "\t ".join([ "%.2f" % (np.mean(logreg_test_eval[name])) for name in metric_names ]))
print("RFC\t\t" + "\t ".join([ "%.2f" % (np.mean(rfc_test_eval[name])) for name in metric_names ]))
print("XGB\t\t" + "\t ".join([ "%.2f" % (np.mean(xgb_test_eval[name])) for name in metric_names ]))
print("GBM\t\t" + "\t ".join([ "%.2f" % (np.mean(gbm_test_eval[name])) for name in metric_names ]))

with open("performance_results", "w") as f:
    f.write("***CROSS-VALIDATION PERFORMANCE***\n")
    f.write("MODEL\t" + "\t ".join(metric_names) + "\n")
    f.write("Rand\t" + "\t ".join([ "%.2f" % (np.mean(random_train_eval[name])) for name in metric_names ]) + "\n")
    f.write("Strat\t" + "\t ".join([ "%.2f" % (np.mean(stratified_train_eval[name])) for name in metric_names ]) + "\n")
    f.write("Minor\t" + "\t ".join([ "%.2f" % (np.mean(minor_train_eval[name])) for name in metric_names ]) + "\n")
    f.write("Major\t" + "\t ".join([ "%.2f" % (np.mean(major_train_eval[name])) for name in metric_names ]) + "\n")
    f.write("Prior\t" + "\t ".join([ "%.2f" % (np.mean(prior_train_eval[name])) for name in metric_names ]) + "\n")
    f.write("LogReg\t" + "\t ".join([ "%.2f" % (np.mean(logreg_train_eval[name])) for name in metric_names ]) + "\n")
    f.write("RFC\t" + "\t ".join([ "%.2f" % (np.mean(rfc_train_eval[name])) for name in metric_names ]) + "\n")
    f.write("XGB\t" + "\t ".join([ "%.2f" % (np.mean(xgb_train_eval[name])) for name in metric_names ]) + "\n")
    f.write("GBM\t" + "\t ".join([ "%.2f" % (np.mean(gbm_train_eval[name])) for name in metric_names ]) + "\n")

    f.write("\n")

    f.write("***TEST PERFORMANCE***\n")
    f.write("MODEL\t" + "\t ".join(metric_names) + "\n")
    f.write("Rand\t" + "\t ".join([ "%.2f" % (np.mean(random_test_eval[name])) for name in metric_names ]) + "\n")
    f.write("Strat\t" + "\t ".join([ "%.2f" % (np.mean(stratified_test_eval[name])) for name in metric_names ]) + "\n")
    f.write("Minor\t" + "\t ".join([ "%.2f" % (np.mean(minor_test_eval[name])) for name in metric_names ]) + "\n")
    f.write("Major\t" + "\t ".join([ "%.2f" % (np.mean(major_test_eval[name])) for name in metric_names ]) + "\n")
    f.write("Prior\t" + "\t ".join([ "%.2f" % (np.mean(prior_test_eval[name])) for name in metric_names ]) + "\n")
    f.write("LogReg\t" + "\t ".join([ "%.2f" % (np.mean(logreg_test_eval[name])) for name in metric_names ]) + "\n")
    f.write("RFC\t" + "\t ".join([ "%.2f" % (np.mean(rfc_test_eval[name])) for name in metric_names ]) + "\n")
    f.write("XGB\t" + "\t ".join([ "%.2f" % (np.mean(xgb_test_eval[name])) for name in metric_names ]) + "\n")
    f.write("GBM\t" + "\t ".join([ "%.2f" % (np.mean(gbm_test_eval[name])) for name in metric_names ]) + "\n")