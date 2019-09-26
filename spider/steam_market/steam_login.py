import requests
import json
import base64
import ssl
import time
import execjs

ssl._create_default_https_context = ssl._create_unverified_context
account = 'win8h'
pw_ = 'eGlhbmppYW5AODYzLmNu'
pw = str(base64.b64decode(pw_),'utf-8')


def get_steam_login_cookie(user,passwd):
    login_cookie = ''
    key_url = 'https://store.steampowered.com/login/getrsakey/'
    login_url = 'https://store.steampowered.com/login/dologin/'
    header = {'Referer': 'https://store.steampowered.com/login/',
              'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like '
                            'Gecko) Chrome/75.0.3770.142 Safari/537.36'
              }
    post_data_rsa = {'donotcache': str(int(time.time()*1000)),
                 'username': user
                 }
    with open('rsa.js', encoding='utf-8') as f:
        rsa_js = f.read()
    try:
        print('正在向服务器获取rsa公钥。。。')
        sess = requests.session()
        page = sess.post(key_url,data = post_data_rsa,headers = header)

    except Exception as e:
        print('获取公钥失败，建议检查vpn是否失效')
        print(e)
        return None
    rsa_data = json.loads(page.text)
    pub_mod = rsa_data.get('publickey_mod')
    pub_exp = rsa_data.get('publickey_exp')
    timestamp = rsa_data.get('timestamp')
    # gid = json_data.get('token_gid')
    passencrypt = execjs.compile(rsa_js).call('getpwd', passwd, pub_mod, pub_exp)
    post_data_login = {
        'donotcache':str(int(time.time()*1000)),
        'username':user,
        'password':passencrypt,
        'twofactorcode':'',
        'emailauth':'',
        'loginfriendlyname':'',
        'captchagid':'-1',
        'captcha_text':'',
        'emailsteamid':'',
        'rsatimestamp':timestamp,
        'remember_login':'true',
    }
    try:
        page = sess.post(login_url,data = post_data_login,headers=header)
    except Exception as e:
        print('第一次提交账户信息失败。一般不会有这事儿。。。你退群吧')
        print(e)
        return None
    json_data = json.loads(page.text)
    if json_data['success']== False and json_data['requires_twofactor']== True and json_data['message']== '':
        print('检测到该账户需要手机令牌')
        post_data_rsa = {'donotcache': str(int(time.time() * 1000)),
                         'username': user
                         }
        try:
            print('正在获取第二次rsa公钥。。。')
            page = sess.post(key_url, data=post_data_rsa,headers = header)
        except Exception as e:
            print('公钥获取失败')
            print(e)
            return None
        rsa_data = json.loads(page.text)
        pub_mod = rsa_data.get('publickey_mod')
        pub_exp = rsa_data.get('publickey_exp')
        timestamp = rsa_data.get('timestamp')
        passencrypt = execjs.compile(rsa_js).call('getpwd', passwd, pub_mod, pub_exp)
        while True:
            phone_guard = input('请输入steam手机令牌：')
            post_data_login = {
                'donotcache': str(int(time.time() * 1000)),
                'username': user,
                'password': passencrypt,
                'twofactorcode': phone_guard,
                'emailauth': '',
                'loginfriendlyname': '',
                'captchagid': '-1',
                'captcha_text': '',
                'emailsteamid': '',
                'rsatimestamp': timestamp,
                'remember_login': 'true',
            }
            print('正在尝试登陆。。。')
            try:
                page = sess.post(login_url,data = post_data_login,headers = header)
                transfer_data = json.loads(page.text).get('transfer_parameters')
                # test1 = sess.post('https://steamcommunity.com/login/transfer', data=transfer_data, headers=header)
                test = sess.post('https://help.steampowered.com/login/transfer', data=transfer_data, headers=header)
            except Exception as e:
                print('提交手机令牌出错。不是，你这破网能不能行了')
                print(e)
                return None

            login_result = json.loads(page.text)
            if login_result.get('success')==True and login_result.get('login_complete')==True:
                print('登录成功！')
                break
            else:
                print('手机令牌错误！')
        temp = page.cookies.get_dict().items()
        for key,value in temp:
            login_cookie=login_cookie+key+'='+value+';'
        temp = test.cookies.get_dict().get('sessionid')
        login_cookie = login_cookie + 'sessionid' + '=' + temp + ';'
    else:
        print('只做了手机令牌登录，别的号先往后稍稍')
    return login_cookie


if __name__=='__main__':

    cookies = get_steam_login_cookie(account,pw)
    attribute = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like '
                               'Gecko) Chrome/75.0.3770.142 Safari/537.36',
                 'cookie':cookies
                 }
    ht = requests.get('https://steamcommunity.com/tradingcards/boostercreator',headers = attribute).text



