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
    eigen_list = np.array(eigen_list)
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
    # return [energy, Kurtosis / len_data, Skewness, Kurtosis_fr / len_fr, Skewness_fr, mean_50, mean_150, class_name]
    return [energy, Kurtosis / len_data,  Kurtosis_fr / len_fr,  mean_50, mean_150, class_name]


# 从这里开始，data_dispose用来删除数据里的时间戳，但有时候处理完还会报错，需要去txt里看看第二行是不是有突出部分
# 数据文件命名统一为0开始，依次累加，如0.txt,1.txt....
# 输入为：文件路径、文件数量。函数中路径都是目标文件的文件夹路径，比如0.txt，路径为xxx/xxx/0.txt，那么输入为xxx/xxx/
# for f_path in path:
#     data_dispose(f_path,5)

# data_divide函数用来切分数据有效部分，参数为输入路径，输出路径，待切分的文件数量
# 使用方法：函数会显示一张数据可视化图片，根据图片需要给出我们认为是有效数据的数据段，
# 如100 500 1000 2000指100-500，1000-2000两个数据段
# 另外，这里至少需要一个数据段；返回的为按段切分好的数据片段文件，切了几段就有几个，最后整合在输出路径下
# 有个没测试的地方，建议使用前先把输出路径文件夹建好，没引入os包，我忘了看这个函数会不会自己建立文件夹了= =
# data_divide('唐森/temp/man/','唐森/disposed/man/',5)

# 这个函数用来生成5维的特征向量，保存为一个Excel
# 参数为：输入路径，输出路径，文件数量（即上一步生成的文件数），标签，多少个数据为一组（用多少个数据算特征，目前为2k），
# 滑动窗口的步长（每隔多少个数据做一次特征计算），最后一个是特征的数量（不含标签）
# 这里的输出路径不一样，给的不是文件夹，而是最终文件路径
generate_eigenvalue('唐森/2019-7-1/','唐森/en3.xls',1,0,2000,500,5)
