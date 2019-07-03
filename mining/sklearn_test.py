# -*- coding: utf-8 -*-
import numpy as np      #导入np方便矩阵运算
import xlrd     #用于读取表格
from sklearn.linear_model import LogisticRegression #从sklearn中导入6个模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings     #由于数据问题，LDA分类会输出warning，暂时屏蔽该提示


#warnings.filterwarnings('ignore') #暂时忽略LDA的warning


def xls_tolist_python(xls):     #定义函数将读取的表格数据转换为list数据
    temp = [[0 for i in range(7)] for j in range(90)]       #建立空list，存放接下来的表格数据
    for i in range(90):     #遍历表格数据行数
        for j in range(7):  #便利表格列数
            temp[i][j] = xls.cell(i + 1, j + 1).value   #读取数据，赋值
    cols = xls.col_values(7)        #最后一列为标签
    for i in range(len(temp)):      #遍历temp行
        if cols[i + 1] == 'yes':        #标签为yes则赋值1，no则赋值0
            temp[i][6] = 1
        else:
            temp[i][6] = 0
    result = np.array(temp)     #最终结果为python的list，将其转化为np的list，方便处理
    return result       #返回list数据


workbook_train = xlrd.open_workbook(r'e:/Learning/Data_113.xls')        #读取113表格
workbook_test = xlrd.open_workbook(r'e:/Learning/Data_114.xls')     #读取114表格
sheet_train = workbook_train.sheet_by_name('sheet1')        #读取113表格第一页
sheet_test = workbook_test.sheet_by_name('sheet1')          #读取114表格第一页
data_train = xls_tolist_python(sheet_train)   #转化为np矩阵形式
data_test = xls_tolist_python(sheet_test)       #转化为np矩阵形式
X_train = data_train[:,0:6] #取前六列作为训练集输入
y_train = data_test[:,6]  #最后一列为标签
X_test = data_test[:,0:6] #取114的前六列做测试

logreg = LogisticRegression()   #实例化对象
logreg.fit(X_train, y_train)    #拟合模型
y_test = logreg.predict(X_test) #利用模型对114进行预测
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

knn = KNeighborsClassifier()#实例化对象
knn.fit(X_train, y_train)#拟合模型
y_test = knn.predict(X_test)#利用模型对114进行预测
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

clf = DecisionTreeClassifier()#实例化对象
clf.fit(X_train, y_train)#拟合模型
clf.predict(X_test)#利用模型对114进行预测
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

lda = LinearDiscriminantAnalysis()#实例化对象
lda.fit(X_train, y_train)#拟合模型
y_test = lda.predict(X_test)#利用模型对114进行预测
print('Accuracy of LDA classifier on training set: {:.2f}'.format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'.format(lda.score(X_test, y_test)))

gnb = GaussianNB()#实例化对象
gnb.fit(X_train, y_train)#拟合模型
y_test = gnb.predict(X_test)#利用模型对114进行预测
print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'.format(gnb.score(X_test, y_test)))

svm = SVC()#实例化对象
svm.fit(X_train, y_train)#拟合模型
y_test = svm.predict(X_test)#利用模型对114进行预测
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))

