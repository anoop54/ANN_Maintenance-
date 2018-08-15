import serial
import sys
import glob
import serial


'''
ser = serial.Serial('/dev/tty.usbserial', 9600)
while True:
    print ser.readline()'''

i = 0
x = []
def main():
   #x = serial_ports()
   #myport = x[0]
    serialread()

    
def serialread():
    global i
    global x 
    ser = serial.Serial('COM12', 9600)
    while True:
        x.insert(i,float(str(ser.readline())))
        i = i+1
        if(i == 256):
            print x
            i= 0

def serialread1(prt):
    ser = serial.Serial(prt, 115200)
    while True:
        print(ser.readline())
              
def serial_ports():
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

main()



