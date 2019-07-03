import requests
from bs4 import BeautifulSoup as bs
import re
import os
import pandas as pd

from requests import exceptions
search_content = ['爬虫','python开发','数据挖掘']
search_url = 'https://www.shixiseng.com/interns/c-110100_?k=%s&t=zh&p='
home_url = 'https://www.shixiseng.com'
attribute = {'user-agent':'Mozilla/5.0'}


def get_page(url,header):
    try:
        r = requests.get(url%search_content[0],headers = header,timeout = 3)
        r.encoding=r.apparent_encoding
        r.raise_for_status()
        return r.text
    except exceptions.HTTPError as e:
        print(e)


def get_info(html):
    position_info = {'job_name':[],
                     'salary':[],
                     'work_time':[],
                     'company':[],
                     'scale':[],
                     'job_detail':[],
                     'location':[],
                     'end_data':[],
                     'upload_data':[]
                     }
    soup = bs(html,features='html.parser')
    job_list = soup.find_all('ul',class_ = 'position_list')
    for jobs in job_list.find_all('li'):
        job_name = jobs.a.string
        job_url = home_url+jobs.a.sttrs['href']
        job_slary = jobs.find_all('span',class_ = 'position-salary').string


def code_trans(s):
    pattern = r'&#x[a-z0-9]{3}'



html_ = get_page(search_url+'1',header=attribute)
soup = bs(html_,features='lxml')
job_list = soup.find_all('ul',class_ = 'position-list')
job = unicode('s')
job_name = job_list[0].find_all('a')[0].string
print(bytes(job_name,encoding='utf-8'))

