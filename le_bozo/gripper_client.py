import socket

def commandGripper(msg):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("172.20.10.3", 3777)) #change this to the ip address
    s.send(bytes(msg,"utf-8"))

if __name__ == "__main__":
    commandGripper("open")
    commandGripper("close")
    commandGripper("calibrate")
    commandGripper("shutoff")