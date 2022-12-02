import socket
from gripper_isolated import * 
import gpiozero as GPIO

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 777)) #change this to the ip address
s.listen()

servo1 = GPIO.AngularServo(17, min_pulse_width = 1/3500, max_pulse_width = 1/1250) #reminder to setup the gripper in these way using M2F wires
servo2 = GPIO.AngularServo(27, min_pulse_width = 1/5500, max_pulse_width = 1/1500)
servo_list = [servo1, servo2]

while True:
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established.")
    msg = clientsocket.recv(64).decode("utf-8")
    match msg:
        case "open":
            print("registered open")
            openGripper(servo2, servo1)
        case "close":
            print("registered close")
            closeGripper(servo2, servo1)
        case "calibrate":
            print("registered calibrate")
            calibrateGripper(servo_list)
        case "shutoff":
            print("registed shutoff")
            break
    clientsocket.close()

print("connection closed")
    