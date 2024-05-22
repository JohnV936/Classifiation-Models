# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:19:51 2024

@author: jverl
"""

import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("D:/Data Science MTU/Final Project/hyperaktiv/Current/EDA/Reproduction/raw_days_all_data.csv", 
                   index_col="ID")

drop_rows = [10,15,30,112,113,164,166,186,221,230,231,240,241,242,
             320,322,348,349,350,351,378,379,382,429,459,461,465,
             505,506,515,517,518,519,520,521,520,521,522,523,524,
             525,526,527,528,529,530,531,532,533,534,535,538,539,
             540,564,565,580,581,582,583,584,585,586,587,618,644,
             645,646,650,651,660,661,662,663,664,665,679,680,681,
             695,696,697,698,699,734,735,736,737,738,739,740,741,
             755,756,757,758,759,760,761,762,832,833,834,835,836,
             837,838,839,840,883,884,885,886,900,901,902,903,904,
             905,906,907,908,909,910,911,912,913,914,915,916,917,
             918,919,920,921,922,923,924,925,926,927,928,929,930,
             940,957,958,959,960,961,962,963,964,965,966,967,968,
             969,970,971,972,973,974,975,976,977,978,992,993,994,
             995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,
             1007,1008,1009,1010,1011,1012,1026,1033,1047,1048,1049,
             1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,
             1061,1062,1063,1064,1065,1078,1079,1080,1081,1082,1083,
             1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,
             1095,1096,1097,1098,1099,1112,1113,1114,1115,1116,1117,
             1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,
             1129,1130,1131,1132,1133,1147,1148,1149,1150,1151,1152,
             1153,1167,1168,1169,1170,1171]

data = data.reset_index()
data = data.drop(drop_rows, axis=0)
data = data.drop("activity__benford_correlation", axis=1)
data = data.set_index("ID")

cats = pd.read_csv("D:/Data Science MTU/Final Project/hyperaktiv/Current/EDA/Reproduction/predict_dataset_with_cat.csv",
                    index_col="ID")

cats = cats[["ADHD", "AGE", "SEX", "MED"]]

cohort_list= []

for p in cats.index:
    p_string = p.split("_")
    adhd = cats["ADHD"][cats.index==p][0]
    if adhd == 1:
        cohort_list.append([p, "ADHD"])
    elif p_string[0] == "control":
        cohort_list.append([p, "control"])
    else:
        cohort_list.append([p, "non_ADHD"])

cohort = pd.DataFrame(cohort_list, columns = ["ID", "Cohort"])
cohort = cohort.set_index("ID")

data = data.join(cats)

data = data.join(cohort)

data = data.reset_index()

data_1 = data.groupby("Cohort").mean(numeric_only=True)
data_1 = data_1.transpose()

data_1.iloc[0:1439].plot(title= "Average Day Plots by Cohort", xlabel ="Minutes", ylabel="Activity")
plt.close("all")


data_2 = data.groupby("ADHD").mean(numeric_only=True)

data_2 = data_2.transpose()

data_2 = data_2.rename(columns={0:"healthy/non-ADHD", 1: "ADHD"})

data_2.iloc[0:1439].plot(title= "Average Day Plots by ADHD", xlabel ="Minutes", ylabel="Activity")

data_3 = data[data["SEX"]==1].groupby(["ADHD"]).mean(numeric_only=True)
data_3 = data_3.transpose()
data_3 = data_3.rename(columns={0:"healthy/non-ADHD", 1: "ADHD"})
data_3.iloc[0:1439].plot(title= "Average Day Plots by ADHD for Male Participants", xlabel ="Minutes", ylabel="Activity")

data_4 = data[data["SEX"]==0].groupby(["ADHD"]).mean(numeric_only=True)
data_4 = data_4.transpose()
data_4 = data_4.rename(columns={0:"healthy/non-ADHD", 1: "ADHD"})
data_4.iloc[0:1439].plot(title= "Average Day Plots by ADHD for Female Participants", xlabel ="Minutes", ylabel="Activity")

data_5 = data[data["AGE"]=="17-29"].groupby(["ADHD"]).mean(numeric_only=True)
data_5 = data_5.transpose()

data_5.plot(title= "Average Day Plots by ADHD Ages 17 - 29")

data_6 = data[data["AGE"]=="30-39"].groupby(["ADHD"]).mean(numeric_only=True)
data_6 = data_6.transpose()

data_6.plot(title= "Average Day Plots by ADHD Ages 30 - 39")

data_7 = data[data["AGE"]=="40-49"].groupby(["ADHD"]).mean(numeric_only=True)
data_7 = data_7.transpose()

data_7.plot(title= "Average Day Plots by ADHD Ages 40 - 49")

data_8 = data[data["AGE"]=="50-69"].groupby(["ADHD"]).mean(numeric_only=True)
data_8 = data_8.transpose()

data_8.plot(title= "Average Day Plots by ADHD Ages 50 - 69")


