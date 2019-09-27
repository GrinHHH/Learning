# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import re
# import requests
# import pybase64
import os
import xlwt
import xlrd
# import fontforge


# data = np.loadtxt('数据/wind/震动传感器大风条件下测试数据/20190311（唐森风)/六号 大风/x.txt')
# count = 10
# for i in range(count):
#     y1 = np.fft.fft(data[(i)*2000:(i+1)*2000])[:1000]
#     fftx = np.linspace(0, 250, 999)
#     plt.plot(fftx,abs(y1[1:1000]))
#     # plt.savefig('image-fr/en/en_%d.jpg'%i)
#     plt.show()
#     plt.close()

def data_dispose(input_path,output_dir = '',output_path='xls',slide_length=2000,interval=500,property_num=5,first_=0,second_=0):
    if not os.path.exists("disposed"):
        os.makedirs("disposed")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if first_ == 0:
        for dirpath, dirnames, filenames in os.walk(input_path):
            if filenames:
                temp = dirpath.split("\\")[-1]
                if not os.path.exists("disposed"+"/"+temp):
                    os.makedirs("disposed"+"/"+temp)
                file_num = 0
                for name in filenames:

                    with open(dirpath+"/"+name,"r+",encoding="utf-8") as f:
                        try:
                            result = []
                            for line in f.readlines():

                                if re.search(r'\t', line):
                                    content = line.split("\t")[0]
                                    content += '\n'
                                else:
                                    content = line
                                result.append(content.strip(b'\x00'.decode()))

                            f.seek(0)
                            f.truncate(0)
                            f.writelines(result)
                        finally:
                            f.close()
                    data = np.loadtxt(dirpath+"/"+name)
                    print(dirpath)
                    plt.plot(data)
                    plt.show()
                    str_in = input('请输入块个数及起始位置：')
                    plt.close()
                    try:
                        position = [int(n) for n in str_in.split(' ')]
                        for j in range(position[0]):
                            slide = data[position[1 + j * 2]:position[2 + j * 2]]
                            np.savetxt("disposed"+"/"+temp+"/"+str(file_num)+".txt", slide)
                            file_num += 1
                    except Exception as e:
                        print(e)
                        continue

    for dirpath, dirnames, filenames in os.walk("disposed"):
        if filenames:
            temp = dirpath.split("\\")[-1]
            class_name = set_class(temp)
            eigen_list = [[0 for x in range(property_num + 1)]]
            for name in filenames:
                data = np.loadtxt(dirpath+"/"+name)
                if second_ != 0:
                    if class_name != 0:
                        for i in range(len(data)):
                            data[i] = data[i]*1.1
                for i in range(int((len(data)-slide_length+interval)/interval)):
                    if (i*interval+slide_length)>len(data):
                        slide = data[(len(data)-slide_length):len(data)]
                        eigenvalue = cal_eigenvalue(slide,class_name)
                        eigen_list.append(eigenvalue)
                        break
                    slide = data[i*interval:(i*interval+slide_length)]
                    eigenvalue = cal_eigenvalue(slide,class_name)
                    eigen_list.append(eigenvalue)
                print ('one done')
            eigen_list = np.array(eigen_list)
            file_excel = xlwt.Workbook()
            file_excel_sheet1 = file_excel.add_sheet(u'sheet1', cell_overwrite_ok=True)
            for rows in range(len(eigen_list)-1):
                for cols in range(property_num+1):
                    file_excel_sheet1.write(rows, cols, float(eigen_list[rows+1][cols]))
            file_excel.save(output_path+"/"+str(temp)+".xls")
            print ('xls done')


def cal_eigenvalue(data, class_name):
    fr = abs(np.fft.fft(data))
    fr = fr[1:1000]
    energy = 0.
    Kurtosis = 0.
    Skewness = 0.
    Kurtosis_fr = 0.
    Skewness_fr = 0.
    fr_50 = fr[0:200]
    fr_150 = fr[200:600]
    len_data = len(data)
    len_fr = len(fr)
    const = float(len_data) / ((len_data - 1) * (len_data - 2))
    const_fr = float(len_fr) / ((len_fr - 1) * (len_fr - 2))
    for i in range(len(data)):
        energy += data[i] * data[i]
    mean = np.mean(data)
    mean_fr = np.mean(fr)
    std = np.std(data)
    for j in range(len_data):
        Kurtosis += np.power((data[j] - mean) / std, 4)
        Skewness += np.power((data[j] - mean), 3)
    std_fr = np.std(fr)
    for j in range(len_fr):
        Kurtosis_fr += np.power((fr[j] - mean_fr) / std_fr, 4)
        Skewness_fr += np.power((fr[j] - mean_fr), 3)
    Skewness = const * Skewness / np.power(std, 3)
    Skewness_fr = const_fr * Skewness_fr / np.power(std_fr, 3)
    mean_50 = np.mean(fr_50)
    mean_150 = np.mean(fr_150)
    # return [energy, Kurtosis / len_data, Skewness, Kurtosis_fr / len_fr, Skewness_fr, mean_50, mean_150, class_name]
    return [energy, Kurtosis / len_data,  Kurtosis_fr / len_fr,  mean_50, mean_150, class_name]


def set_class(string):
    if "car" in string or "车" in string:
        return 2
    elif "man" in string or "人" in string:
        return 1
    else:
        return 0


data_dispose('E:\测试',first_=1,second_=1)
