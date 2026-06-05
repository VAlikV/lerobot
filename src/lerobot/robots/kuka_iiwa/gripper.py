import serial


class Gripper:
    def __init__(self, device="/dev/ttyACM0", baudrate=115200, timeout=1):
        self._serial = serial.Serial(device, baudrate, timeout=timeout)
        self._is_open = True

    def send(self, state):
        """state > 0 - open, state <= 0 - close"""
        if state <= 0:
            self._serial.write(b'Close\n')
            self._is_open = False
        else:
            self._serial.write(b'Open\n')
            self._is_open = True

    @property
    def is_open(self):
        return self._is_open

    def close(self):
        if self._serial.is_open:
            self._serial.close()