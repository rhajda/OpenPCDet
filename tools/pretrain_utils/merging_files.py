import os
import sys
import shutil


####
####
srcpath = '\simulation\01_no_noise\pcd_sim_noise_001\train\label'
savepath = '\for_training\sim_no_noise\label_2'
data_typ = '.txt'     # for labels use '.txt' and for point clouds use '.pcd'
####
####


src_day1 = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20211209' + data
src_day2 = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20211210' + data
src_day3 = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20220103' + data
src_day4 = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20220106' + data
src_day5 = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20220107' + data

#Download the pointclouds of each day
os.chdir(src_day1)
filesday1 = os.listdir()
os.chdir(src_day2)
filesday2 = os.listdir()
os.chdir(src_day3)
filesday3 = os.listdir()
os.chdir(src_day4)
filesday4 = os.listdir()
os.chdir(src_day5)
filesday5 = os.listdir()

#Training
num = 0
src = src_day1 + srcpath + '\0'
dst = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data' + savepath + '\0'
for file in filesday1:
    numstr = str(num).zfill(5)
    shutil.copyfile(src = src + numstr + data_typ, dst = dst + numstr + data_typ)
    num = num + 1

num = 0
new_num = num + len(filesday1)
src = src_day2 + srcpath + '\0'
dst = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data' + savepath + '\0'
for file in filesday2:
    numstr = str(num).zfill(5)
    new_numstr = str(new_num).zfill(5)
    shutil.copyfile(src = src + numstr + data_typ, dst = dst + new_numstr + data_typ)
    num = num + 1
    new_num = new_num + 1

num = 0
src = src_day3 + srcpath + '\0'
dst = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data' + savepath + '\0'
for file in filesday3:
    numstr = str(num).zfill(5)
    new_numstr = str(new_num).zfill(5)
    shutil.copyfile(src = src + numstr + data_typ, dst = dst + new_numstr + data_typ)
    num = num + 1
    new_num = new_num + 1

num = 0
src = src_day4 + srcpath + '\0'
dst = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data' + savepath + '\0'
for file in filesday4:
    numstr = str(num).zfill(5)
    new_numstr = str(new_num).zfill(5)
    shutil.copyfile(src = src + numstr + data_typ, dst = dst + new_numstr + data_typ)
    num = num + 1
    new_num = new_num + 1

num = 0
src = src_day5 + srcpath + '\0'
dst = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data' + savepath + '\0'
for file in filesday5:
    numstr = str(num).zfill(5)
    new_numstr = str(new_num).zfill(5)
    shutil.copyfile(src = src + numstr + data_typ, dst = dst + new_numstr + data_typ)
    num = num + 1
    new_num = new_num + 1
