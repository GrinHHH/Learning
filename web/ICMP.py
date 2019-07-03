# -*- coding: utf-8 -*-
import socket
import os
import struct
import time
import select

ICMP_ECHO_REQUEST = 8
ICMP_ECHO_CODE = 0


def cal_checksum(str):
    csum = 0
    countTo = (len(str) / 2) * 2
    count = 0
    while count < countTo:
        thisVal = str[count + 1] * 256 + str[count]
        csum = csum + thisVal
        csum = csum & 0xffffffff
        count = count + 2
    if countTo < len(str):
        csum = csum + str[len(str) - 1].decode()
        csum = csum & 0xffffffff
    csum = (csum >> 16) + (csum & 0xffff)
    csum = csum + (csum >> 16)
    answer = ~csum
    answer = answer & 0xffff
    answer = answer >> 8 | (answer << 8 & 0xff00)
    return answer


class Ping:
    def __init__(self,address,num):

        self.address = socket.gethostbyname(address)
        self.num = num
        self.timeout = 1
        self.processId = os.getpid() & 0xFFFF

    def send_echo(self,my_socket,sequence):
        checksum = 0
        icmp_header = struct.pack("!bbHHh", ICMP_ECHO_REQUEST, ICMP_ECHO_CODE, checksum, self.processId, sequence)
        data = struct.pack('!d',time.time())
        checksum = cal_checksum(icmp_header+data)
        icmp_header = struct.pack("!bbHHh", ICMP_ECHO_REQUEST, ICMP_ECHO_CODE, checksum, self.processId, sequence)
        my_socket.sendto((icmp_header+data),(self.address,1))

    def recv_echo(self,my_socket,order):
        whatReady = select.select([my_socket], [], [], self.timeout)
        if not whatReady[0]:  # Timeout
            return None
        result = my_socket.recvfrom(1024)[0]
        time_recv = time.time()
        echo_header = result[20:28]
        type, code, checksum, packetID, sequence = struct.unpack("!bbHHh", echo_header)
        if type == 0 and packetID == self.processId and sequence == order:
            send_time_len = struct.calcsize('!d')
            send_time = struct.unpack('!d',result[28:28+send_time_len])[0]
            delay = time_recv - send_time
            ttl = ord(struct.unpack("!c", result[8:9])[0].decode())
            return delay, ttl, send_time_len

    def ping(self):
        print("Pinging " + self.address + " using Python:")
        print("")
        loss = 0
        for i in range(self.num):
            mySocket = socket.socket(socket.AF_INET, socket.SOCK_RAW, 1)
            self.send_echo(mySocket,i)
            result = self.recv_echo(mySocket,i)
            mySocket.close()
            if not result:
                print("Request time out")
                loss += 1
            else:
                delay = int(result[0] * 1000)
                ttl = result[1]
                bytes = result[2]
                print(
                    "Received from " + self.address + ": byte(s)=" + str(bytes) + " delay=" + str(delay) + "ms TTL=" + str(ttl))
            time.sleep(1)  # one second
        print("Packet: sent = " + str(self.num) + " received = " + str(self.num - loss) + " lost = " + str(loss))


ex = Ping('www.baidu.com',4)
ex.ping()
