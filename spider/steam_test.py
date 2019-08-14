import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import re
import json

# proxy = '198.18.0.3:443'
# proxies = {'http':'http://'+proxy,
#          'https':'https://'+proxy}

cookie = '_ga=GA1.2.1118973353.1532583290; timezoneOffset=28800,0; steamMachineAuth76561198383121964=7752' \
         '8A42EE0FD319F77A2B2D5B81361020546B63; steamMachineAuth76561198159269171=67483656A28C580D92599703' \
         '03A28A3E8ECADF31; browserid=1189412587868647152; _gid=GA1.2.891359323.1565614878; steamRememberL' \
         'ogin=76561198159269171%7C%7C3700d6c6cead9dede0630a9bf37bb54c; strInventoryLastContext=753_6; ses' \
         'sionid=62f9a2858b7b275e371f9fcc; steamLoginSecure=76561198159269171%7C%7CC7D220DD1A31BE08819854D' \
         'D318A568C369F6390; webTradeEligibility=%7B%22allowed%22%3A1%2C%22allowed_at_time%22%3A0%2C%22ste' \
         'amguard_required_days%22%3A15%2C%22new_device_cooldown_days%22%3A7%2C%22time_checked%22%3A1565675612%7D'

url = 'https://steamcommunity.com/tradingcards/boostercreator'
url_gems = 'https://steamcommunity.com/market/listings/753/753-Sack%20of%20Gems'
attribute = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like '
                          'Gecko) Chrome/75.0.3770.142 Safari/537.36'
             ,'cookie':cookie
            }


def get_page(urls,header):
    sess = requests.session()
    try:
        r = sess.get(urls,headers = header)
        r.encoding = 'utf-8'
        r.raise_for_status()
        return r.text
    except Exception as e:
        return 'Error Occurred,%s' % e


def get_content(html):
    pattern = r'\[\{[\S\s]*?\]'
    temp = re.findall(pattern,html)
    pattern_ = r'\{[\S\s]*?\}'
    game_list = re.findall(pattern_, temp[0])
    if not game_list:
        return '获取游戏列表失败'
    else:
        index = []
        df_temp = [[]]
        count = 0
        for rows in game_list:
            game_info = json.loads(rows)
            if count == 0:
                index = game_info.keys()
            df_temp.append(game_info.values())
        df_game = pd.DataFrame(df_temp[1:],columns=index)
        return df_game


def get_price(urls,header):

    item_page = get_page(urls, header=header)
    pattern = r'Market_LoadOrderSpread\(\s(\S*)\s?\)'
    item_id = re.findall(pattern, item_page)
    js_frame = get_page('https://steamcommunity.com/market/itemordershistogram?cou''ntry=CN&language=schinese&currenc'
                        'y=23&item_nameid=%s&two_factor=0'%item_id[0], header=attribute)
    pattern_amt = r'<td align=\\"right\\">(\S*)?<\\/td>'
    pattern_cost = r'<td align=\\"right\\" class=\\"\\">\\u00a5\s(\S*)?<\\/td>'
    amt = re.findall(pattern_amt,js_frame)
    cost = re.findall(pattern_cost,js_frame)
    amt = list(map(int,amt))
    cost = list(map(float,cost))
    price = round((cost[0]*amt[0]+cost[1]*amt[1])/(amt[0]+amt[1])*0.98,2)
    return price


# def get_item_id(urls,header):
#     item_page = get_page(urls,header=header)
#     pattern = r'Market_LoadOrderSpread\(\s(\S*)\s?\)'
#     item_id = re.findall(pattern,item_page)
#     if item_id is not None:
#         return item_id[0]
#     else:
#         return '获取物品id失败'


def cal_income(game_info):
    list_flag = []
    list_income = []
    price_gems = get_price(url_gems,attribute)
    for cols,rows in game_info.iterrows():
        flag = ''
        name = rows['name']
        appid = rows['appid']
        price = rows['price']
        url_pack = 'https://steamcommunity.com/market/listings/753/' + '%s-%s' % (appid,name) + '%20Booster%20Pack'
        try:
            price_cash = get_price(url_pack,attribute)
            income = (price_cash*0.85-int(price)*1.0/1000*price_gems*0.8)*0.85
            print('游戏名称：' + name + '  ' + '每日理论收入' + str(income))
            if income < 0:
                flag = '+'
            else:
                flag = '-'
            income = str(abs(income))
        except Exception as e:
            print(e)
            flag = 'Nan'
            income = 'Nan'
        list_flag.append(flag)
        list_income.append(income)
    game_info.insert(1,'flag',list_flag)
    game_info.insert(1,'income',list_income)
    return game_info


if __name__ == '__main__':
    # page = get_page(url,header=attribute)
    # game = get_content(page)
    # result = cal_income(game)
    # print(result.info)
    # result.to_csv('income.csv')
    df = pd.read_csv('income.csv')
    x = df.query('flag=="-"')
    df_ = x['income']
    s = 0.
    for ele in df_:
        s = float(ele)+s

    print (s)








