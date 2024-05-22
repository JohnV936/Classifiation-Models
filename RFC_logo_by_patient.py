###############################################################################################################
# This code reads in the cleaned tsfresh features and the labels. It removes columns with all 
# missing values and columns with no variance. It assigns a group number for each participant
# then uses these to set up the Leave-One-Group-Out Cross Validation along with a Random Forest Classifier.
# In this version, the 6 best features based on EDA are selected.
# The code can, and has been, easily adapted to perform various other experiments.
# The results are then aggregated by participant
############################################################################################################### 

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler

def cut_rows(data_by_day, patient_info, max_days):
    patient_id = list(patient_info.index)
    day_list = []
    
    for patient in patient_id:
        day_list.append(data_by_day[data_by_day.index==patient]) 

    cut_day_list = []
    for df in day_list:
        if df.shape[0] >=max_days:
            df_cut = df[:max_days]
            cut_day_list.append(df_cut)
        else:
            cut_day_list.append(df)
    cut_data_by_day = pd.concat(cut_day_list, axis=0)
    return cut_data_by_day


# Dropping columns that have na values
def clean_missing_columns(data_by_day):
    for column in data_by_day.columns:
        if data_by_day[column].isnull().sum(axis=0) > 0:
            data_by_day = data_by_day.drop([column], axis=1)
    return data_by_day

def variance_thresh(data_by_day, thresh):
    var_thr = VarianceThreshold(threshold=thresh)
    
    data_by_day_X = data_by_day.drop("ADHD", axis=1)
    data_by_day_Y = data_by_day["ADHD"]
    var_thr.fit(data_by_day_X)

    selected = data_by_day_X.columns[var_thr.get_support()]
    data_by_day_X = data_by_day[selected]
    data_by_day = pd.concat([data_by_day_X, data_by_day_Y], axis=1)
    return data_by_day

def cohort_filter(data_by_day, unwanted):
    drops = []
    for name in data_by_day.index:
         cohort = name.split("_")[0]
         if cohort == unwanted:
             drops.append(name)
    data_by_day = data_by_day.drop(drops, axis=0)
    return data_by_day    

def grouping(data_by_day, patient_info):
    patient_info["group"] = -1
    i = 0
    for index in patient_info.index:
        patient_info.at[index, "group"] = i
        i+=1
    data_by_day = data_by_day.join(patient_info["group"])
    return data_by_day

def aggregate_score(results, thresh):
    predicted = results["Predicted"]
    true = results["True"]
    
    predicted_df = pd.DataFrame(predicted)
    predicted_df = predicted_df.set_index(results_df.index)
    predicted_df = predicted_df.groupby(predicted_df.index).mean(numeric_only=True)

    for index in predicted_df.index:
        if predicted_df["Predicted"][index] >= thresh:
            predicted_df["Predicted"][index] = 1
        else:
            predicted_df["Predicted"][index]= 0


    true_grouped = true.groupby(true.index).median(numeric_only=True)

    metrics = precision_recall_fscore_support(true_grouped, 
                                              predicted_df,
                                              pos_label=1,
                                              average="binary")
    return metrics

def reduce_dimensions_2(data, thres):
    correlations = data.corr().abs()
    mask = np.triu(np.ones_like(correlations, dtype=bool))

    tri_df = correlations.mask(mask)
    correlations = correlations.reset_index()
    to_drop = [c for c in tri_df.columns if any(tri_df[c]>thres)]
    
    data = data.drop(to_drop, axis=1)
    return data

data_by_day = pd.read_csv("D:\Data Science MTU\Final Project\hyperaktiv\Current\Feature Extraction\Hyperaktiv\hyper_features_clean.csv", index_col="ID")
data_by_day = data_by_day.drop("col1", axis=1)

# These are the 6 selected features from EDA
selected = ['activity__fft_coefficient__attr_"abs"__coeff_36',
        'activity__fft_coefficient__attr_"abs"__coeff_35',
        'activity__fft_coefficient__attr_"angle"__coeff_33',
        'activity__fft_coefficient__attr_"angle"__coeff_15',
        'activity__fft_coefficient__attr_"angle"__coeff_38',
        'activity__fft_coefficient__attr_"angle"__coeff_53'
        ]
data_by_day = data_by_day[selected]
patient_info = pd.read_csv("D:\Data Science MTU\Final Project\hyperaktiv\Current\Provided data\patient_info.csv", index_col="ID", delimiter=";")
patient_info = patient_info[patient_info["ACC"]==1]
labels = patient_info["ADHD"]

data_by_day = data_by_day.join(labels)

data_by_day = grouping(data_by_day, patient_info)

data_by_day = clean_missing_columns(data_by_day)

data_by_day = variance_thresh(data_by_day, 0)

data_by_day = data_by_day.rename(columns = lambda x:re.sub('"', '', x))
data_by_day = data_by_day.rename(columns = lambda x:re.sub(',', '', x))

#data_by_day = cut_rows(data_by_day, patient_info, 1)

groups = data_by_day["group"]
data_by_day = data_by_day.drop(["group"], axis=1)
dataX = data_by_day.drop(["ADHD"], axis=1)
dataY = data_by_day["ADHD"]

logo = LeaveOneGroupOut()

clf = RandomForestClassifier(n_estimators=1000, ccp_alpha=0.01, max_samples=0.3)
# 0.53

features = []
pred_list = []
true_list = []
prob_list = []
indexes = []

# Perform cross-validation
for train_index, test_index in logo.split(dataX, dataY, groups=groups):
    X_train, X_test = dataX.iloc[train_index], dataX.iloc[test_index]
    y_train, y_test = dataY.iloc[train_index], dataY.iloc[test_index]
     
    # Fit model on training data
    
    model = clf.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:,1]
    
    #x = data_aggX.iloc[data_aggX.index ==X_test.index[0]]
    #pred = model.predict(x)
    #prob = model.predict_proba(x)[:,1]
    pred = model.predict(X_test)
    print("="*20)
    print("patient " + str(y_test.index[0]))
    print("testing:")
    print(pred)
    #print(data_aggY.iloc[data_aggY.index == y_test.index[0]].values)
    print(y_test.values)
    print("="*10)
    print("training accuracy")
    print(accuracy_score(y_train, model.predict(X_train)))
    print("training precision, recall, F")
    print(precision_recall_fscore_support(y_train, model.predict(X_train), 
                                          pos_label=1 ,
                                          average="binary"))
    
    length = X_test.shape[0]
    
    for i in range(length):
        pred_list.append(pred[i])
        prob_list.append(prob[i])
        true_list.append(y_test.values[i])
        indexes.append(X_test.index[i])

    #features.append(model.feature_importances_)

#plot_tree(clf)

results_dict = {"Predicted":pred_list,
                "True": true_list,
                "Prob": prob_list}

results_df = pd.DataFrame(results_dict, index=indexes)

results_df["Prob"].plot(kind="hist", title="Histogram of Probabilities for Phase 2 Experiments")

aggregate_score(results_df, 0.5)
precision_recall_fscore_support(results_df["True"] , results_df["Predicted"])

f = []

for i in range(len(features)):
    y = pd.Series(features[i], index=dataX.columns)
    f.append(y)

