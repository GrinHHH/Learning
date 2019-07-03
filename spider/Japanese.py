# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import os

header = {'user-agent':'Mozilla/5.0'}


# 以下为py3.6版本，open方法可以设置为encoding=utf-8，此时传入数据须为str格式
def get_content(filename, urls):
    # urls = str(urls)
    page_branch = requests.get(urls,headers = header)
    page_branch.encoding = 'utf-8'
    temp_branch = page_branch.text
    codes = BeautifulSoup(temp_branch,features='html.parser')
    content = codes.find_all('ul',class_='fxxx')[0].text
    if not os.path.exists('data'):
        os.makedirs('data')
    if os.path.exists('data/%s.txt' % filename):
        os.remove('data/%s.txt' % filename)
    with open('data/%s.txt' % filename, 'a',encoding='utf-8')as f:
        f.write(content)
        f.close()


r = requests.get('http://www.v5jp.com/html/fuxi/fuxi.html#n3yflk',headers = header)
r.encoding = 'utf-8'
page = r.text
html = BeautifulSoup(page,features='html.parser')
rdlist = html.find_all('div',class_='fxlist')
temp = rdlist[1].encode('utf-8')
rd = BeautifulSoup(temp,features='html.parser')
reading = rd.find_all('a')
for lines in reading:
    if lines.text.find('N2') == -1:
        continue
    else:
        get_content(lines.text,lines.get('href'))


# 以下为py2.7版本，open方法自身不带encode参数，若要写utf-8，则必须保证文件名、内容在操作之前全部为utf-8编码
# def get_content(filename, urls):
#     # urls = str(urls)
#     page_branch = requests.get(urls)
#     page_branch.encoding = 'utf-8'
#     temp_branch = page_branch.text
#     codes = BeautifulSoup(temp_branch,features='html.parser')
#     content = codes.find_all('ul',class_='fxxx')[0].text
#     if os.path.exists('data/%s.txt' % filename):
#         os.remove('data/%s.txt' % filename)
#     with open('data/%s.txt'.encode('utf-8') % filename, 'a')as f:
#         f.write(content.encode('utf-8'))
#         f.close()
#
#
# r = requests.get('http://www.v5jp.com/html/fuxi/fuxi.html#n3yflk')
# r.encoding = 'utf-8'
# page = r.text
# html = BeautifulSoup(page,features='html.parser')
# rdlist = html.find_all('div',class_='fxlist')
# temp = rdlist[1].encode('utf-8')
# rd = BeautifulSoup(temp,features='html.parser')
# reading = rd.find_all('a')
# for lines in reading:
#     if lines.text.find('N3') == -1:
#         continue
#     else:
#         get_content(lines.text,lines.get('href'))
