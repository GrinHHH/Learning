import re
import numpy as np
import xlwt
from matplotlib import pyplot as plt

# path_i = ['唐森/temp/car/','唐森/temp/man/','唐森/temp/en2/']
# path_o = ['唐森/disposed/car/','唐森/disposed/man/','唐森/disposed/en2/']
path = {'in':['唐森/temp/car/','唐森/temp/man/','唐森/temp/en2/'],
        'num':[5,5,5],
        'out':['唐森/disposed/car/','唐森/disposed/man/','唐森/disposed/en2/']}


def data_dispose(input_path,num):
    for i in range(num):
        title_txt = open('%s%d.txt'%(input_path,i), 'r+')
        try:
            full_txt = title_txt.readlines()
            temp = np.shape(full_txt)[0]
            regex = "\t"
            new_txt = []
            x = 0
            for line in full_txt:
                if re.search(regex, line):
                    trans = ''
                    list_1 = list(line)
                    list_2 = ['0','0','0','0','0','0','0','0','0','0','0']
                    for j in range(10):
                        list_2[j] = list_1[j]

                    if x != (temp-1):
                        list_2[10] = "\n"
                    else:
                        list_2[10] = ''
                    line = trans.join(list_2)
                    x += 1
                new_txt.append(line)

                # else:
                    # new_txt.append(line)

            title_txt.seek(0)
            title_txt.truncate(0)
            title_txt.writelines(new_txt)
        finally:
            title_txt.close()
        print('over')


def data_divide(input_path,output_path,num):
    file_num = 0
    for i in range(num):
        data = np.loadtxt(input_path+str(i)+'.txt')
        plt.plot(data)
        plt.show()
        str_in = input('请输入块个数及起始位置：')
        plt.close()
        position = [int(n) for n in str_in.split(' ')]
        for j in range(position[0]):
            slide = data[position[1+j*2]:position[2+j*2]]
            np.savetxt(output_path+str(file_num)+'.txt',slide)
            file_num += 1


def generate_eigenvalue(input_path,output_path,file_num,class_name,slide_length,interval,property_num):
    eigen_list = [[0 for x in range(property_num+1)]]
    for files in range(file_num):
        data = np.loadtxt(input_path+'%d.txt' % files)
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
    file_excel = xlwt.Workbook()
    file_excel_sheet1 = file_excel.add_sheet(u'sheet1', cell_overwrite_ok=True)
    for rows in range(len(eigen_list)-1):
        for cols in range(property_num+1):
            file_excel_sheet1.write(rows, cols, float(eigen_list[rows+1][cols]))
    file_excel.save(output_path)
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
    return [energy, Kurtosis / len_data, Skewness, Kurtosis_fr / len_fr, Skewness_fr, mean_50, mean_150, class_name]


# for f_path in path:
#     data_dispose(f_path,5)
# data_divide('唐森/temp/man/','唐森/disposed/man/',5)
# generate_eigenvalue('唐森/disposed/en2/','唐森/en2.xls',5,0,2000,500,7)
# data_dispose('',2)
# data = np.loadtxt('1.txt')
generate_eigenvalue('','唐森/en_.xls',1,0,2000,500,7)