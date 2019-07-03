# -*- coding: utf-8 -*-
from socket import *


proxyPort = 800
proxyIp = '10.51.244.187'
proxyServerSocket = socket(AF_INET, SOCK_STREAM)
proxyServerSocket.bind((proxyIp, proxyPort))
proxyServerSocket.listen(5)
while True:
    print 'Proxy is ready'
    proxyConnectSocket, clientAddr = proxyServerSocket.accept()
    print 'Connection Confirmed'
    message = proxyConnectSocket.recv(1024)
    print 'message recv'
    filename = message.split()[1].partition("/")[2].partition('/')[2]

    try:

        f = open(filename,'rb')
        print 'Catch exist,transferring'
        output = f.read()
        header = ' HTTP/1.1 200 OK\nConnection: close\nContent-Type: text/html\nContent-Length: %d\n\n' % (
            len(output))
        proxyConnectSocket.send(header.encode())
        proxyConnectSocket.send(output.encode())
        proxyConnectSocket.close()
        print 'Transfer ends'
    #在此进行数据传输，等会来补
    except IOError:
        print 'No such catch,requesting host'
        hostName = message.split()[1].partition("/")[2].partition("/")[0]
        hostIp = gethostbyname(hostName)
        proxyClientSocket = socket(AF_INET, SOCK_STREAM)
        try:
            proxyClientSocket.connect((hostIp, 80))
            proxyClientSocket.send(message.replace('110.51.244.187:800',hostName))
            proxyBuf = proxyClientSocket.recv(10240)
            print 'Recv from host'
            proxyConnectSocket.sendall(proxyBuf)  # 不一定能传，记得改
            print ' Send over'
            proxyConnectSocket.close()
            # x = open(filename, 'w')
            # x.write(proxyBuf)
            # 这里要写把文件保存下来
        except:
            header = 'HTTP/1.1 404 Not Found'
            proxyConnectSocket.send(header.encode())
            proxyConnectSocket.close()
        proxyClientSocket.close()
