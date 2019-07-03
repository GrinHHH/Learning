# -*- coding: utf-8 -*-
from socket import *
import random


protocol_type = raw_input('which type of protocol do you want?')
if protocol_type == 'tcp':
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.bind(('10.51.244.187', 8000))
    serverSocket.listen(2)
    print 'Hello~ here is Papa`s server'
    while True:
        connectionSocket, addr = serverSocket.accept()
        consult_message = connectionSocket.recv(1024)
        temp = consult_message.split()
        if consult_message.find('PS4') != -1:
            reply_message = 'Come on Boy,you`ll get one finally'
        elif consult_message.find('bye') != -1:
            reply_message = 'Good Night my Son'
        else:
            reply_message = 'Papa is busy now,go and find your Mom,alright?'
        connectionSocket.send(reply_message)
        connectionSocket.close()
elif protocol_type == 'udp':
    serverSocket = socket(AF_INET,SOCK_DGRAM)
    serverSocket.bind(('10.51.244.187',8000))
    while True:
        print 'UDP test is beginning'
        rand = random.randint(0, 10)
        message, addr = serverSocket.recvfrom(1024)
        message = message.upper()
        if rand < 4:
            continue
        serverSocket.sendto(message,addr)
else:
    print'Sorry,there is no such service'
