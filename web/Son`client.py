# -*- coding: utf-8 -*-
from socket import *
import time

server_name = '10.51.244.187'
server_port = 8000
protocol_type = input('Which protocol do you want?')
if protocol_type == 'tcp':
    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((server_name, server_port))
    while True:
        message = input('What do you want to say to Papa?')
        clientSocket.send(message)
        result = clientSocket.recv(1024)
        print ('Papa`s answer is :' + result)
        if result.find('Good Night') != -1:
            clientSocket.close()
            break
    print ('You are disconnected with Papa')
elif protocol_type == 'udp':
    clientSocket = socket(AF_INET,SOCK_DGRAM)
    clientSocket.settimeout(1)
    lossCount = 0.
    for i in range(10):
        sendTime = time.time()
        message = ('Ping %d %s' % (i + 1, sendTime)).encode()
        try:
            clientSocket.sendto(message,(server_name,server_port))
            recMessage,server_addr = clientSocket.recvfrom(1024)
            rtt = time.time() - sendTime
            print('Sequence %d: Reply from %s    RTT = %.3fs' % (i + 1, server_name, rtt))
        except Exception as e:
            print('Sequence %d: Request timed out' % (i + 1))
            lossCount += 1
    print ('For here is inner net,it`s useless to cal RTT')
    print ('Loss Rate: %.2f'%(lossCount/10))
    clientSocket.close()
