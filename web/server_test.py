# -*- coding: utf-8 -*-
#import socket module
from socket import *
from urllib.parse import quote,unquote

serverSocket = socket(AF_INET, SOCK_STREAM)
#Prepare a sever socket
serverSocket.bind(('10.51.180.209',800))
serverSocket.listen(5)
# path = bytes('D:/','utf-8')
path = 'D:/'
while True:
    # Establish the connection
    print ('Ready to serve...')
    connectionSocket, addr = serverSocket.accept()
    try:
        message = connectionSocket.recv(1024)
        temp = message.split()
        if len(temp) <= 0:
            header = 'HTTP/1.1 204 N0 Content'
            connectionSocket.send(header.encode())
            connectionSocket.close()
        else:
            filename = bytes.decode(message.split()[1])
            name = unquote(filename[1:])
            # print(name)
            f = open(path+name,'rb')
            outputdata = f.read()
            #Send one HTTP header line into socket
            header = ' HTTP/1.1 200 OK\nConnection: close\nContent-Type: text/html' \
                     '\nContent-Length' \
                     ': %d\n\n' % (len(outputdata))
            connectionSocket.send(header.encode())
            #Send the content of the requested file to the client

            connectionSocket.send(outputdata)
            connectionSocket.close()
    except IOError:
        #Send response message for file not found
        header = 'HTTP/1.1 404 Not Found'
        connectionSocket.send(header.encode())
        #Close client socket
        connectionSocket.close()
serverSocket.close()