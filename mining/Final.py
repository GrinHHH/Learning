# -*- coding: utf-8 -*-
import pandas as pd # 导入pandas读取txt，这里numpy读取速度较慢，故换用pd
import numpy as np # 利用np的list可以简化许多计算
import pydotplus # 决策树可视化辅助包
import xlwt # 表格操作包，用于将特征写入表格
import xlrd # 表格读取包，用于读取表格
import pywt # 小波变换包，用于小波分解
from scipy import signal # 用于做带通滤波器
import matplotlib.pyplot as plt # 画图用
from sklearn import tree # 导入决策树模型
from sklearn.model_selection import train_test_split # 利用改函数划分数据集
from sklearn.metrics import * # 可以用该包导出模型的评价
import os # 决策树可视化Graphviz的环境变量没用，因此直接在工程导入
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


propertyNum = 10    #最终采用的特征数量
dataLen = 2830      #数据长度
fs = 200  # 设立频率变量fs
lowcut = 0.5    #低通频率
highcut = 30    #高通频率
order = 5  # 设立滤波器阶次变量
nyq = 0.5 * fs  # 设立采样频率变量nyq，采样频率=fs/2。
low = lowcut / nyq
high = highcut / nyq
b, a = signal.butter(order, [low, high], btype='band')  # 设计巴特沃斯带通滤波器 “band”
lv = 4 # 小波滤波层数
m = 2 # 起始层数
n = 4 # 结束层数
wavefunc = 'db4'#小波变换方式


def cal_zcr(zcr): #定义函数计算过零率，输入数据帧
    num_zcr = 0.    #计数变量
    for position_zcr in range(len(zcr)-1): #循环遍历
        if zcr[position_zcr]*zcr[position_zcr+1] < 0: #过零一次计数器加一
            num_zcr += 1
    return num_zcr/len(zcr) #返回过零率


def xls_tolist_python(xls,property_num,data_len):     #定义函数将读取的表格数据转换为list数据，输入表格，列数，行数
    temp = [[0 for i in range(property_num)] for j in range(data_len)]       #建立空list，存放接下来的表格数据
    for rows in range(dataLen):     #遍历表格数据行数
        for cols in range(property_num):  #便利表格列数
            temp[rows][cols] = xls.cell(rows , cols ).value   #读取数据，赋值
    result = np.array(temp)     #最终结果为python的list，将其转化为np的list，方便处理
    return result       #返回list数据


def filter_by_wavelet(plt_data): #定义函数执行小波滤波，输入为数据帧
    coeff = pywt.wavedec(plt_data, wavefunc, mode='sym', level=lv, axis=0) #将滤波后的小波数据放在coeff中
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # 非线性函数，用于收缩数据值
    for i in range(m, n + 1):  # 选取小波系数层数为 m~n层，尺度系数不需要处理
        cD = coeff[i]   # 分解后的小波系数
        for j in range(len(cD)): #遍历cd
            Tr = np.sqrt(2 * np.log(len(cD)))  # 计算阈值
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) - Tr  # 向零收缩
            else:
                coeff[i][j] = 0  # 低于阈值置零
    denoised_index = pywt.waverec(coeff, wavefunc) #反变换回时域
    return denoised_index


def plot_wave(data, start_position, end_position, sleep_status, frequency, use_band,use_wavelet,number):#画图函数，输入数据整体，起始位置，结束位置，标签，后三个置一时执行相应操作，0时不执行
    temp = data[3000*start_position:3000*end_position]  #取数据
    data_temp = []
    for bug in range(len(temp)): #将数据由pandas多维数组转化为一维
        data_temp.append(temp[bug][0])
    data_temp = np.array(data_temp)
    # nmb pandas读完每行都是一个list，怪不得老子fft出来长得都一样
    if use_band == 1: #为1执行带通滤波
        data_temp = signal.lfilter(b, a, data_temp)
    if use_wavelet == 1: #为1执行小波滤波
        data_temp = filter_by_wavelet(data_temp)
    if frequency: #为1执行快速傅里叶变换
        y = np.fft.fft(data_temp)[:len(data_temp)/2]
        fftx = np.linspace(0, 100, len(data_temp)/2)
        plt.figure() #画图
        plt.plot(fftx,abs(y))
        if use_band == 1: #根据不同参数保存的位置及命名均不同
            plt.savefig(r'sleep_classify/frequency/band_status_frequency_%d_%s.png' % (sleep_status, number))
        elif use_wavelet == 1:
            plt.savefig(r'sleep_classify/frequency/wavelet_status_frequency_%d_%s.png' % (sleep_status, number))
        else:
            plt.savefig(r'sleep_classify/frequency/status_frequency_%d_%s.png' % (sleep_status, number))
    else: #不执行频域转换
        plt.figure() #画图
        plt.plot(data_temp)
        if use_band ==1:#不同参数路径不同
            plt.savefig(r'sleep_classify/band_status_%d_%s.png' % (sleep_status, number))
        elif use_wavelet ==1:
            plt.savefig(r'sleep_classify/wavelet_status_%d_%s.png' % (sleep_status, number))
        else:
            plt.savefig(r'sleep_classify/status_%d_%s.png' % (sleep_status, number))
    # plt.show()
    plt.close('all') #关闭画图


def generate_property_xls(property_num,data_len,use_band,use_wavelet,file_name): #用于从数据生成特征表格，输入为特征数量，数据总量长度，及是否过滤参数，及要保存的文件名
    data = pd.read_table('sc4002e0_data.txt', header=None) #利用pandas读取数据
    data = np.array(data) #转化为nparray
    label = pd.read_table('sc4002e0_label.txt', header=None) #读标签
    label = np.array(label)
    data_attribute = [[0 for i in range(property_num)] for j in range(data_len)]#建立矩阵保存特征
    for rows in range(data_len): #遍历数据
        temp = data[rows * 3000:rows * 3000 + 3000] #3000个一帧
        slide = []#因为pandas的list问题，要进行一次转换，同上
        for temp_rows in range(len(temp)):
            slide.append(temp[temp_rows][0])
        if use_wavelet == 1: #为一做小波滤波
            slide = filter_by_wavelet(slide)
        if use_band == 1: #为1做带通滤波
            slide = signal.lfilter(b, a, slide)
        data_max = np.max(slide) #求最大值
        data_min = np.min(slide)#最小值
        data_range = data_max - data_min#极差
        data_var = np.var(slide)#方差
        data_zcr = cal_zcr(slide)#过零率
        coeff_1 = pywt.wavedec(slide, wavefunc, mode='sym', level=7, axis=0)#由于元祖不可复制，又需要5个，故只好多建几个
        coeff_2 = pywt.wavedec(slide, wavefunc, mode='sym', level=7, axis=0)#每个coeff存放7层小波变换后的值
        coeff_3 = pywt.wavedec(slide, wavefunc, mode='sym', level=7, axis=0)
        coeff_4 = pywt.wavedec(slide, wavefunc, mode='sym', level=7, axis=0)
        coeff_5 = pywt.wavedec(slide, wavefunc, mode='sym', level=7, axis=0)
        cD_6 = np.var(pywt.waverec(trasform(coeff_1, 2), 'db4', axis=0)) #提取每层小波变换，将其他层置0后反变换，并计算方差
        cD_5 = np.var(pywt.waverec(trasform(coeff_2, 3), 'db4', axis=0))
        cD_4 = np.var(pywt.waverec(trasform(coeff_3, 4), 'db4', axis=0))
        cD_3 = np.var(pywt.waverec(trasform(coeff_4, 5), 'db4', axis=0))
        cD_2 = np.var(pywt.waverec(trasform(coeff_5, 6), 'db4', axis=0))
        data_attribute[rows] = [data_max, data_min, data_range, data_var, data_zcr,#特征写入矩阵中
                                cD_6,cD_5,cD_4,cD_3,cD_2,label[rows]]
    print 'Done'
    file_excel = xlwt.Workbook() #建立表格
    file_excel_sheet1 = file_excel.add_sheet(u'sheet1', cell_overwrite_ok=True)#sheet1
    for rows in range(data_len): #遍历矩阵，将数值写入表格
        for cols in range(property_num):
            file_excel_sheet1.write(rows, cols, float(data_attribute[rows][cols]))
    file_excel.save(r'sleep_classify/data_%s.xls'%file_name)#保存


def cal_type_amount(label):#用于计算划分数据集后每个集中的标签数量，输入为标签list
    amount = [0 for i in range(7)]
    label_type = [0,1,2,3,4,5,6]#7种标签
    for rows in range(len(label)):#遍历数据集标签
        for j in range(7):
            if label[rows] == label_type[j]:#相应类别+1
                amount[j] += 1
    return amount #返回数量矩阵


def SampEn(U, m, r): #计算样本熵函数，这里并不是自己完成的，因此有待优化
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        B = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1.0) / (N - m) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(B)

    N = len(U)

    return -np.log(_phi(m + 1) / _phi(m))


def trasform(data,rows): #将小波变换后除想要的层之外的数据置0，用于反变换，输入为变换后的小波分解数据和想要保留的层数
    for i in range(len(data)): #遍历层数
        for j in range(len(data[i])):#遍历层中数据
            if i!=rows:#不想保留则置零
                data[i][j]=0
    return data#返回处理后的数据


def plot_wavelet(data,start_position,end_position,number): #用于画图观察小波变换的结果
    temp = data[start_position * 3000:end_position * 3000] #具体操作同plot_wave相同，这里就不再解释了
    plot_data = []
    for rows in range(len(temp)):
        plot_data.append(temp[rows][0])
    plot_data = filter_by_wavelet(plot_data)
    coeff_1 = pywt.wavedec(plot_data, wavefunc, mode='sym', level=7, axis=0)
    coeff_2 = pywt.wavedec(plot_data, wavefunc, mode='sym', level=7, axis=0)
    coeff_3 = pywt.wavedec(plot_data, wavefunc, mode='sym', level=7, axis=0)
    coeff_4 = pywt.wavedec(plot_data, wavefunc, mode='sym', level=7, axis=0)
    coeff_5 = pywt.wavedec(plot_data, wavefunc, mode='sym', level=7, axis=0)
    cD_6 = pywt.waverec(trasform(coeff_1, 2), 'db4', axis=0)
    cD_5 = pywt.waverec(trasform(coeff_2, 3), 'db4', axis=0)
    cD_4 = pywt.waverec(trasform(coeff_3, 4), 'db4', axis=0)
    cD_3 = pywt.waverec(trasform(coeff_4, 5), 'db4', axis=0)
    cD_2 = pywt.waverec(trasform(coeff_5, 6), 'db4', axis=0)
    plt.figure()
    plt.subplot(5,1,1)
    plt.plot(cD_6)
    plt.subplot(5, 1, 2)
    plt.plot( cD_5)
    plt.subplot(5, 1, 3)
    plt.plot(cD_4)
    plt.subplot(5, 1, 4)
    plt.plot( cD_3)
    plt.subplot(5, 1, 5)
    plt.plot( cD_2)
    plt.savefig(r'sleep_classify/status_%d_3.png'%number)
    #plt.show()
    plt.close('all')


def plot_group(data,group,use_filter,use_wavelet,name):#用于将7类数据放在一起观察，输入为原始数据
    # 想要看的数据位置（数组），以及是否执行滤波变量，保存的文件名，操作同单个画图
    for rows in range(len(group)):
        temp = data[3000 * group[rows][0]:3000 * group[rows][1]]
        data_temp = []
        for bug in range(len(temp)):
            data_temp.append(temp[bug][0])
        data_temp = np.array(data_temp)
        if use_filter == 1:
            data_temp = signal.lfilter(b, a, data_temp)
        if use_wavelet == 1:
            data_temp = filter_by_wavelet(data_temp)
        plt.subplot(len(group),1,rows+1)
        plt.plot(data_temp)
    plt.savefig(r'sleep_classify/group_%s'% name)
    plt.close()

# 画图部分
# group = [[700,701],[1516,1517],[875,876],[1285,1286],[1656,1657],[1740,1741],[936,937]] #想要可视化数据帧的序号
# data = pd.read_table('sc4002e0_data.txt', header=None) #画图时需要单独读取数据
# data = np.array(data)
# label = pd.read_table('sc4002e0_label.txt', header=None)
# label = np.array(label)
# for rows in range(len(group)): #循环画图
#     plot_wave(data, group[rows][0]+2,group[rows][1]+2,label[group[rows][0]],frequency=1,use_band=0,use_wavelet=1,number=3)

#数据处理部分
#generate_property_xls(propertyNum,dataLen,use_band= 0,use_wavelet=1,file_name='final_wavelet')#首先生成特征表格
print 'Start Fit'
workbook = xlrd.open_workbook(r'sleep_classify/data_final_var.xls')#读取生成的表格
sheet_train = workbook.sheet_by_name('sheet1')
x = xls_tolist_python(sheet_train,propertyNum,dataLen)#将表格转换为nparray，保存在x中
overSampling = np.array([x[936] for i in range(50)])#样本不平衡，执行过采样
x = np.concatenate((x,overSampling),axis=0)#过采样数据加入x

# 训练部分
X,Y = train_test_split(x,test_size=0.25)#划分数据集
X_train = X[:,0:propertyNum-1]#训练集
X_label = X[:,-1]#训练集标签
clf = tree.DecisionTreeClassifier(max_depth=10)#加载决策树模型，设定最大层数为10
clf.fit(X_train,X_label)#训练模型
dot_data = tree.export_graphviz(clf, out_file=None)#将训练的模型保存

# 可视化部分，可直接屏蔽
graph = pydotplus.graph_from_dot_data(dot_data) #读取保存的模型，可视化
graph.write_png("tree.png")#可视化决策树保存

# 结果输出
types = clf.predict(Y[:,0:propertyNum-1]) #验证集预测
pos = 0.#定义计数变量
for num in range(len(Y)):#计算预测准确率
    if Y[num][-1]==types[num]:
       pos+=1
print '精确度：'
print pos/len(Y)
print '训练集标签数量统计：'
print cal_type_amount(X_label)
print '测试集标签数量统计：'
print cal_type_amount(Y[:,-1])
print '预测标签数量统计：'
print cal_type_amount(types)
Matrix = confusion_matrix(Y[:,-1],types)#混淆矩阵
print Matrix
report = classification_report(Y[:,-1],types) #模型评估
print report