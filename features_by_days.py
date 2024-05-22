# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:33:17 2024

@author: jverl
"""
##############################################################################################
# This code takes in the motor activity timeseries csv files for both the Hyperaktiv dataset
# and the Healthy Controls from the Psykose/Depresjon Dataset
# It also, retrieves information for each patient on their age, sex, medication, and whether
# have ADHD. These were used later in the EDA.
# For each timeseries, each day is extracted seperately into a dataframe. Incomplete days
# are discarded. 
# Tsfresh extracts its features for every full day and these form the rows of the dataset.
##############################################################################################


import os
import glob
import pandas as pd
from tsfresh import extract_features

# Get CSV files list from a hyperaktiv dataset 
path_hyper = "D:\\Data Science MTU\\Final Project\\hyperaktiv\\Dataset\\activity_data"
csv_files_hyper = glob.glob(path_hyper + "/*.csv")

# Get CSV files list from psykose dataset
path_psyk = "D:\\Data Science MTU\\Final Project\\Psykose\\control"
csv_files_psyk = glob.glob(path_psyk + "/*.csv")

# loading patient_info to a dataframe
patient_info_hyper = pd.read_csv("D:\\Data Science MTU\\Final Project\\hyperaktiv\\Dataset\\patient_info.csv",delimiter=";" )

# filtering for people who wore actigraph
patient_info_hyper_ACC = patient_info_hyper[patient_info_hyper["ACC"]==1]
#patient_info_hyper_ACC["ACC"]
#patient_info_hyper_ACC.columns

patient_info_depresjon = pd.read_csv("D:\\Data Science MTU\\Final Project\\Depresjon\\scores.csv")
patient_info_control = patient_info_depresjon[23:56]    

# converting age intervals to age codes in line with hyperaktiv study 
patient_info_control["age"][patient_info_control["age"] == "20-24"] = 1 # hyperaktiv study starts from 17
patient_info_control["age"][patient_info_control["age"] == "25-29"] = 1
patient_info_control["age"][patient_info_control["age"] == "30-34"] = 2
patient_info_control["age"][patient_info_control["age"] == "35-39"] = 2
patient_info_control["age"][patient_info_control["age"] == "40-44"] = 3
patient_info_control["age"][patient_info_control["age"] == "45-49"] = 3
patient_info_control["age"][patient_info_control["age"] == "50-54"] = 4
patient_info_control["age"][patient_info_control["age"] == "55-59"] = 4
patient_info_control["age"][patient_info_control["age"] == "60-64"] = 4
patient_info_control["age"][patient_info_control["age"] == "65-69"] = 4 # outside of range from hyperaktiv study "50-67" 

# cpnverting gender code in control to allign with hyperaktiv
patient_info_control["gender"][patient_info_control["gender"] == 1] = 0
patient_info_control["gender"][patient_info_control["gender"] == 2] = 1


all_days = []
features_by_day = pd.DataFrame()
adhd_list = []
adhd_df = pd.DataFrame()
age_list = []
age_list_df = pd.DataFrame()
gender_list = []
gender_list_df = pd.DataFrame()
med_list = []
med_list_df = pd.DataFrame()
########################################################################################
# extracting info from hyperaktiv dataset
########################################################################################

# given a filepath and a patient id, returns the timeseries as a dataframe
def read_activity_file(filepath, patient_id):
    if patient_id.split("_")[0] == "patient":
        delimiter= ";"
    if patient_id.split("_")[0] == "control":
        delimiter = "," 
    data = pd.read_csv(filepath, delimiter=delimiter)
    col_to_drop = "date"
    if col_to_drop in data.columns:
        data = data.drop(columns = col_to_drop)
    data.columns=["timestamp", "activity"]
    data["timestamp"]=pd.to_datetime(data["timestamp"]) # convert time strings to pandas datetime
    data["ID"] = patient_id # adding patient ID for traceability during testing
    return data

def get_day_frames(data):
    data = data.set_index("timestamp")
    day_frames = [group[1] for group in data.groupby(data.index.date)] # https://stackoverflow.com/questions/21605491/how-to-split-a-pandas-dataframe-or-series-by-day-possibly-using-an-iterator
    full_day_frames = []
    for day in day_frames:
        if day.shape[0] >= 1440:  
            full_day_frames.append(day)
    
    return full_day_frames


# Looping through all 85 csv files in hyperaktiv dataset
for filepath in csv_files_hyper:
    print("reading: ", filepath)
    patient_id = os.path.splitext(os.path.basename(filepath))[0] # extracts patient id from filepath
    # extract id number from filepath
    # to be used to retrieve patient info from patient_info_hyper_ACC
    record_id = int(patient_id.split("_")[2]) 
    data = read_activity_file(filepath, patient_id) # saving timeseries
    patient_days = get_day_frames(data)
    for day in patient_days:
        day["ID"] = patient_id
        data_features =  extract_features(day, column_id="ID", column_kind=None, column_value="activity")
        all_days.append(data_features)
    
###############################################################################################    
# extracting info from psykose dataset
###############################################################################################

for filepath in csv_files_psyk:
    print("reading: ", filepath)
    patient_id = os.path.splitext(os.path.basename(filepath))[0]
    data = read_activity_file(filepath, patient_id) # saving timeseries
    data = read_activity_file(filepath, patient_id) # saving timeseries
    patient_days = get_day_frames(data)
    for day in patient_days:
        day["ID"] = patient_id
        data_features =  extract_features(day, column_id="ID", column_kind=None, column_value="activity")
        all_days.append(data_features)

features = pd.concat(all_days)

features.to_csv("D:\\Data Science MTU\\Final Project\\hyperaktiv\\Dataset\\features_by_day.csv")


###############################################################################################
# Creating dataframe with target (adhd/no adhd)
###############################################################################################
# Looping through all 85 csv files in hyperaktiv dataset

for filepath in csv_files_hyper:
    print("reading: ", filepath)
    patient_id = os.path.splitext(os.path.basename(filepath))[0] # extracts patient id from filepath
    # extract id number from filepath
    # to be used to retrieve patient info from patient_info_hyper_ACC
    record_id = int(patient_id.split("_")[2])
    
    # finding patient info
    # ADHD
    get_adhd = patient_info_hyper_ACC["ADHD"][patient_info_hyper_ACC["ID"]==record_id].values[0]
    # Age
    get_age = patient_info_hyper_ACC["AGE"][patient_info_hyper_ACC["ID"]==record_id].values[0]
    # gender
    get_gender = patient_info_hyper_ACC["SEX"][patient_info_hyper_ACC["ID"]==record_id].values[0]
    # medication
    get_med = patient_info_hyper_ACC["MED"][patient_info_hyper_ACC["ID"]==record_id].values[0]
    
    adhd_list.append([patient_id, get_adhd])
    age_list.append([patient_id, get_age])
    gender_list.append([patient_id, get_gender])
    med_list.append([patient_id, get_med])
    
    
###############################################################################################    
# extracting info from psykose dataset
###############################################################################################

for filepath in csv_files_psyk:
    print("reading: ", filepath)
    patient_id = os.path.splitext(os.path.basename(filepath))[0]
    
    get_age = patient_info_control["age"][patient_info_control["number"]==patient_id].values[0]
    
    get_gender = patient_info_control["gender"][patient_info_control["number"]==patient_id].values[0]
    
    adhd_list.append([patient_id, 0]) # don't have adhd
    age_list.append([patient_id, get_age])
    gender_list.append([patient_id, get_gender])
    med_list.append([patient_id, 0]) # not on medication

age_list_df = pd.DataFrame(age_list, columns=["ID", "AGE"])
gender_list_df = pd.DataFrame(gender_list, columns=["ID", "SEX"])
med_list_df = pd.DataFrame(med_list, columns=["ID", "MED"])
adhd_df = pd.DataFrame(adhd_list , columns=["ID", "ADHD"])

features = pd.read_csv("D:\\Data Science MTU\\Final Project\\hyperaktiv\\Dataset\\features_by_day.csv")

predict_dataset_2 = features.merge(right=age_list_df, on="ID")
predict_dataset_2 = predict_dataset_2.merge(right=gender_list_df , on="ID")
predict_dataset_2 = predict_dataset_2.merge(right=med_list_df , on="ID")
predict_dataset_2 = predict_dataset_2.merge(right=adhd_df, on="ID")

predict_dataset_2.to_csv("D:\\Data Science MTU\\Final Project\\hyperaktiv\\Dataset\\predict_by_day.csv")

