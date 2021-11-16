import socket
import struct

class UDPClient:
    def __init__(self,addr,port):
        self.addr = (addr,port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_floats(self,buffer):
        buf = struct.pack('%sf' % len(buffer), *buffer)
        self.socket.sendto(buf,self.addr)

    def close(self):
        self.socket.detach()
        self.socket.close()


