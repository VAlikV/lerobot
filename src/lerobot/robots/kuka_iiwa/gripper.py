import serial
from pynput import keyboard

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

def main():
    gripper = Gripper(device="/dev/ttyACM0")

    print("Keyboard control: o - open, c - close, q/esc - quit")

    def on_press(key):
        try:
            if key.char == "o":
                gripper.send(1)
                print("Open")
            elif key.char == "c":
                gripper.send(0)
                print("Close")
            elif key.char == "q":
                return False
        except AttributeError:
            if key == keyboard.Key.esc:
                return False

    try:
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    finally:
        gripper.close()


if __name__ == "__main__":
    main()
