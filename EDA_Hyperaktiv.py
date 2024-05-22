# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:19:51 2024

@author: jverl
"""

import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("D:\\Data Science MTU\\Final Project\\hyperaktiv\\Current\\EDA\\Hyperaktiv\\raw_days_hyper.csv", 
                   index_col="ID")

drop_rows = [1,2,10,15,18,23,24,27,28,30,33,34,37,38,42,43,44,45,47,48,
             112,113,124,133,134,164,166,185,186,221,230,231,236,237,
             240,242,265,275,320,321,322,347,348,349,350,351,378,379,
             382,429,459,461,465]

data = data.reset_index()
data = data.drop(drop_rows, axis=0)
data = data.set_index("ID")

patient_info = pd.read_csv("D:\\Data Science MTU\\Final Project\\hyperaktiv\\Current\\Provided data\\patient_info.csv",
                   delimiter=";",
                   index_col="ID")

patient_info = patient_info[patient_info["ACC"]==1]

patient_info["ADHD"] = patient_info["ADHD"].replace({1:"ADHD", 0:"Non-ADHD"})
patient_info["SEX"] = patient_info["SEX"].replace({0:"female", 1:"male"})
patient_info["AGE"] = patient_info["AGE"].replace({1:"17-29", 2:"30-39",3:"40-49",4:"50-67"})

data = data.reset_index()
for p in data["ID"]:
    p_string = p.split("_")
    num = int(p_string[2])
    #print(num)
    data = data.replace({p:num})

data = data.set_index("ID")
    
data = data.join(patient_info)

data = data.reset_index()
data = data.rename({"ADHD": "Group"}, axis=1)

x = data.iloc[:,1:1441]
 
data_1 = data.groupby("Group").mean(numeric_only=True)
data_1 = data_1.iloc[:,1:1441].transpose()

data_1.plot(title= "Average Day Plots by ADHD", xlabel ="Minutes", ylabel="Activity")

data_2 = data[data["SEX"]=="male"].groupby(["Group"]).mean(numeric_only=True)
data_2 = data_2.iloc[:,1:1441].transpose()

data_2.plot(title= "Average Day Plots by ADHD for Males", xlabel ="Minutes", ylabel="Activity")

data_3 = data[data["SEX"]=="female"].groupby(["Group"]).mean(numeric_only=True)
data_3 = data_3.iloc[:,1:1441].transpose()

data_3.plot(title= "Average Day Plots by ADHD for Females", xlabel ="Minutes", ylabel="Activity")


data_4 = data[data["AGE"]=="17-29"].groupby(["Group"]).mean(numeric_only=True)
data_4 = data_4.iloc[:,1:1441].transpose()

data_4.plot(title= "Average Day Plots by ADHD for Ages 17 - 29")

data_5 = data[data["AGE"]=="30-39"].groupby(["Group"]).mean(numeric_only=True)
data_5 = data_5.iloc[:,1:1441].transpose()

data_5.plot(title= "Average Day Plots by ADHD for Ages 30-39", xlabel ="Minutes", ylabel="Activity")

data_6 = data[data["AGE"]=="40-49"].groupby(["Group"]).mean(numeric_only=True)
data_6 = data_6.iloc[:,1:1441].transpose()

data_6.plot(title= "Average Day Plots by ADHD for Ages 40 - 49")

data_7 = data[data["AGE"]=="50-67"].groupby(["Group"]).mean(numeric_only=True)
data_7 = data_7.iloc[:,1:1441].transpose()

data_7.plot(title= "Average Day Plots by ADHD for Ages 50 - 67")

