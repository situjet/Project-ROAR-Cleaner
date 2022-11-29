import time
import gpiozero as GPIO

def openGripper(leftservo, rightservo):
    #open the gripper
    leftservo.min()
    rightservo.max()
    return 0

def closeGripper(leftservo, rightservo):
    #close the gripper
    leftservo.max()
    rightservo.min()
    return 0

def calibrateGripper(servos):
    #calibrate the gripper
    for servo in servos:
        servo.mid()
    return 0

if __name__ == "__main__":

    #tune these values
    servo1 = GPIO.AngularServo(17, min_pulse_width = 1/3500, max_pulse_width = 1/1250) #reminder to setup the gripper in these way using M2F wires
    servo2 = GPIO.AngularServo(27, min_pulse_width = 1/5500, max_pulse_width = 1/1500)
    servo3 = GPIO.Servo(22,initial_value=None)
    servo_list = [servo1, servo2]
    time.sleep(5)
    print("attempting to calibrate")
    calibrateGripper(servo_list)
    time.sleep(5)
    print("attempting to open")
    openGripper(servo2, servo1)
    time.sleep(5)
    print("attempting to close")
    closeGripper(servo2, servo1)
    time.sleep(5)