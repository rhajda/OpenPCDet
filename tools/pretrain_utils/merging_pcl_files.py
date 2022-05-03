import os
import sys
import shutil

src_day1 = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20211209\simulation\03_noise_005\pcd_sim_noise_005\train\pcl'
src_day2 = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20211210\simulation\03_noise_005\pcd_sim_noise_005\train\pcl'
src_day3 = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20220103\simulation\03_noise_005\pcd_sim_noise_005\train\pcl'
src_day4 = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20220106\simulation\03_noise_005\pcd_sim_noise_005\train\pcl'
src_day5 = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20220107\simulation\03_noise_005\pcd_sim_noise_005\train\pcl'

#Punktewolken der einzelnen Tage hochladen
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

#Ornderstrucktur erstellen
src_real = r"Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\sim\noise_005"

os.chdir(src_real)
os.makedirs('training')
os.chdir(src_real + '\\training')
os.makedirs('calib')
os.makedirs('velodyne')
os.makedirs('label_2')
os.chdir(src_real)
os.makedirs('testing')
os.chdir(src_real + '\\testing')
os.makedirs('calib')
os.makedirs('velodyne')
os.makedirs('label_2')

#Training
num = 0
src = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20211209\simulation\03_noise_005\pcd_sim_noise_005\train\pcl\0'
dst = r"Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\sim\noise_005\indy_sim\training\velodyne\0"
for file in filesday1:
    numstr = str(num).zfill(5)
    shutil.copyfile(src = src + numstr + ".pcd", dst = dst + numstr + ".pcd")
    num = num + 1

num = 0
new_num = num + len(filesday1)
src = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20211210\simulation\03_noise_005\pcd_sim_noise_005\train\pcl\0'
dst = r"Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\sim\noise_005\indy_sim\training\velodyne\0"
for file in filesday2:
    numstr = str(num).zfill(5)
    new_numstr = str(new_num).zfill(5)
    shutil.copyfile(src = src + numstr + ".pcd", dst = dst + new_numstr + ".pcd")
    num = num + 1
    new_num = new_num + 1

num = 0
src = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20220107\simulation\03_noise_005\pcd_sim_noise_005\train\pcl\0'
dst = r"Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\sim\noise_005\indy_sim\training\velodyne\0"
for file in filesday5:
    numstr = str(num).zfill(5)
    new_numstr = str(new_num).zfill(5)
    shutil.copyfile(src = src + numstr + ".pcd", dst = dst + new_numstr + ".pcd")
    num = num + 1
    new_num = new_num + 1

#Validation
num = 0
src = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20220106\simulation\03_noise_005\pcd_sim_noise_005\train\pcl\0'
dst = r"Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\sim\noise_005\indy_sim\training\velodyne\0"
for file in filesday4:
    numstr = str(num).zfill(5)
    new_numstr = str(new_num).zfill(5)
    shutil.copyfile(src = src + numstr + ".pcd", dst = dst + new_numstr + ".pcd")
    num = num + 1
    new_num = new_num + 1

#Testing
num = 0
src = r'Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\20220103\simulation\03_noise_005\pcd_sim_no_noise\train\pcl\0'
dst = r"Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\sim\noise_005\indy_sim\testing\velodyne\0"
for file in filesday3:
    numstr = str(num).zfill(5)
    shutil.copyfile(src = src + numstr + ".pcd", dst = dst + numstr + ".pcd")
    num = num + 1