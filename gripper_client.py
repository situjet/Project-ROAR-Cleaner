import socket

def commandGripper(msg):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("192.168.73.72", 777)) #change this to the ip address
    s.send(bytes(msg,"utf-8"))

if __name__ == "__main__":
    commandGripper("open")
    commandGripper("close")
    commandGripper("calibrate")
    commandGripper("shutoff")