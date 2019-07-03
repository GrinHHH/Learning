import requests
from bs4 import BeautifulSoup as bs
import re


cookie = '3220180608_HEER_LASTACCESS_COOK=; JSESSIONID=yjs1app2~DC8BAD903348B0335280E974B9DCFB75; SECURITY_AUTHE' \
         'NTICATION_COOKIE=086d7778663dad4399814abda59f3b839a3e8951c7b296d0f1e547188ce14dd5670da5c035c4534e; yuns' \
         'uo_session_verify=12cf40ccfe7face95c5ac3a6f8722b9b; __utma=158042917.768424293.1543841358.1545814016.154' \
         '7738239.3; __utmz=158042917.1547738239.3.3.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20p' \
         'rovided); SECURE_AUTH_ROOT_COOKIE=086d7778663dad4399814abda59f3b839a3e8951c7b296d0f1e547188ce14dd5670da5c' \
         '035c4534e'
header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/7'
                       '4.0.3729.169 Safari/537.36',
          'Cookie':cookie}

url = 'http://grdms.bit.edu.cn/yjs/application/main.jsp'
rule = r'charset={\w|\d}'
def get_content(page_url,attr):
    r = requests.get(page_url,headers = attr)
    # encode_type = re.findall()
    # r.encoding = r.apparent_encoding
    soup = bs(r,features='html.parser')
    return soup


page = get_content(url,header)
print (page)
