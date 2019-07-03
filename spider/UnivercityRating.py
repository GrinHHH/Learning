# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup as bs
import bs4
import pandas as pd
import xlwt
import os
url = 'http://www.zuihaodaxue.com/zuihaodaxuepaiming2019.html'
header = {'user-agent':'Mozilla/5.0'}


def get_page(url):
    try:
        r = requests.get(url,headers = header)
        r.encoding = 'utf-8'
        print ('Page Got')
        r.raise_for_status()
        return r.text
    except:
        return 'Error Occurred'


def content_process(html):
    index = []
    result = [[]]
    for rows in html.thead.find_all('th'):
        if rows.contents[0]=="指标得分":
            for elements in html.thead.find_all('option'):
                index.append(elements.attrs['title'])
        else:
            index.append(rows.string)
    for tags in html.tbody.find_all('tr'):
        school = []
        for child in tags.contents:
            if isinstance(child,bs4.element.Tag):
                school.append(child.string)
        result.append(school)
    df_ = pd.DataFrame(result[1:],columns=index)
    df_ = df_.set_index('排名')
    return df_


if __name__ == '__main__':
    soup = bs(get_page(url),features='html.parser')
    content = content_process(soup)
    print(content.head())
    if os.path.exists('output19.xls'):
        os.remove('output19.xls')
    content.to_excel('output19.xls',sheet_name='Sheet_name_1')
    print('xls Done')
