# -*- coding: utf-8 -*-
#import socket module
from socket import *


serverSocket = socket(AF_INET, SOCK_STREAM)
#Prepare a sever socket
serverSocket.bind(('10.106.20.113',80))
serverSocket.listen(5)

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
            filename = message.split()[1]
            f = open(filename[1:],'rb')
            outputdata = f.read()
            #Send one HTTP header line into socket
            header = ' HTTP/1.1 200 OK\nConnection: close\nContent-Type: image/jpeg\nContent-Length: %d\n\n' % (len(outputdata))
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