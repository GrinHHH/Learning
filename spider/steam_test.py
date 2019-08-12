import requests
# from requests import session
from bs4 import BeautifulSoup as bs
import re
proxy = '198.18.0.3:443'
url = 'https://steamcommunity.com/tradingcards/boostercreator'
attribute = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like '
                          'Gecko) Chrome/75.0.3770.142 Safari/537.36'
            }
proxies = {'http':'http://'+proxy,
         'https':'https://'+proxy}
# sess = requests.session()
r = requests.get(url,headers = attribute,proxies = proxies)
r.encoding = 'utf-8'
soup = r.text
page = bs(soup,features='html.parser')
