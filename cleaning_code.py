# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:33:17 2024

@author: jverl
"""
##############################################################################################
# Break down of activity by hour of day
# Comparing adhd, add and control
##############################################################################################


import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

features = pd.read_csv("D:\\Data Science MTU\\Final Project\\hyperaktiv\\Dataset\\features_by_day.csv", index_col="ID")

# Get CSV files list from a hyperaktiv dataset 
path_hyper = "D:\\Data Science MTU\\Final Project\\hyperaktiv\\Dataset\\activity_data"
csv_files_hyper = glob.glob(path_hyper + "/*.csv")

# Get CSV files list from psykose dataset
path_psyk = "D:\\Data Science MTU\\Final Project\\Psykose\\control"
csv_files_psyk = glob.glob(path_psyk + "/*.csv")

# loading patient_info to a dataframe
all_days = []
index = []
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
    data = read_activity_file(filepath, patient_id) # saving timeseries
    patient_days = get_day_frames(data)
    for day in patient_days:
       index.append(patient_id) 
       all_days.append(day["activity"].values)
    

day_df = pd.DataFrame(all_days, index=index)
day_df = pd.concat([day_df,features["activity__benford_correlation"]],axis=1)

#day_df.to_csv("day_plots_benford.csv")



for i in range(day_df.shape[0]):
    x = day_df.iloc[i][0:1440].plot()
    plt.show()
    
features.set_index("ID")

data = pd.read_csv("D:\\Data Science MTU\\Final Project\\hyperaktiv\\The Method\\Working\\No Control\\all_days_values.csv", index_col="ID")

data_categories = pd.read_csv("D:\\Data Science MTU\\Final Project\\hyperaktiv\\The Method\Working\\No Control\\patient_info.csv", 
                              index_col="ID", 
                              delimiter=";")
data_categories = data_categories[data_categories["ACC"]==1]

labels = data_categories["ADHD"]      

# drop_rows = [10,15,30,112,113,164,166,186,221,230,231,240,241,242,
#              320,322,348,349,350,351,378,379,382,429,459,461,465,
#              505,506,515,517,518,519,520,521,520,521,522,523,524,
#              525,526,527,528,529,530,531,532,533,534,535,538,539,
#              540,564,565,580,581,582,583,584,585,586,587,618,644,
#              645,646,650,651,660,661,662,663,664,665,679,680,681,
#              695,696,697,698,699,734,735,736,737,738,739,740,741,
#              755,756,757,758,759,760,761,762,832,833,834,835,836,
#              837,838,839,840,883,884,885,886,900,901,902,903,904,
#              905,906,907,908,909,910,911,912,913,914,915,916,917,
#              918,919,920,921,922,923,924,925,926,927,928,929,930,
#              940,957,958,959,960,961,962,963,964,965,966,967,968,
#              969,970,971,972,973,974,975,976,977,978,992,993,994,
#              995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,
#              1007,1008,1009,1010,1011,1012,1026,1033,1047,1048,1049,
#              1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,
#              1061,1062,1063,1064,1065,1078,1079,1080,1081,1082,1083,
#              1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,
#              1095,1096,1097,1098,1099,1112,1113,1114,1115,1116,1117,
#              1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,
#              1129,1130,1131,1132,1133,1147,1148,1149,1150,1151,1152,
#              1153,1167,1168,1169,1170,1171]

for i in range(data.shape[0]):
    p = data.index[i]
    p_num = int(p.split("_")[2])
    l = labels[p_num]
    title = str(p) + "    label: " + str(l) + "    row: " + str(i)
    x = data.iloc[i][0:1440].plot(title=title)
    plt.show()
    
drop_rows = [1,2,10,15,18,23,24,27,28,30,33,34,37,38,42,43,44,45,47,48,
             112,113,124,133,134,164,166,185,186,221,230,231,236,237,
             240,242,265,275,320,321,322,347,348,349,350,351,378,379,
             382,429,459,461,465]

features = pd.read_csv("D:\\Data Science MTU\\Final Project\\hyperaktiv\\The Method\\Working\\No Control\\features_by_day_2.csv", index_col="ID")

features.iloc[0]


day_features_dirty = features.iloc[drop_rows]
features = features.reset_index()
day_features_clean = features.drop(drop_rows, axis=0)

day_features_clean.to_csv("day_features_clean.csv", index="ID")
day_features_dirty.to_csv("day_features_dirty.csv", index="ID")
