import time
import sys
import win32pipe, win32file, pywintypes
import re

# Record start time
start_time = time.time()
# Counter for None/empty responses
none_count = 0

class PipeClient:
    def __init__(self, pipe_name=r'\\.\pipe\HQ_server'):
        self.pipe_name = pipe_name
        self.handle = None
        self.connected = False

    def connect(self):
        """Establish connection to the named pipe"""
        try:
            self.handle = win32file.CreateFile(
                self.pipe_name,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None
            )
            res = win32pipe.SetNamedPipeHandleState(self.handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
            self.connected = True
            print("Connected to pipe server")
            return True
        except pywintypes.error as e:
            if e.args[0] == 2:
                print("No pipe found, waiting for server...")
                time.sleep(1)
            else:
                print(f"Connection error: {e}")
            return False

    def send_command(self, command):
        """Send a command and receive response"""
        if not self.connected:
            if not self.connect():
                return None

        try:
            # Send the command
            print(f"Sending command: {command}")
            input_data = str.encode(command + '\n')
            win32file.WriteFile(self.handle, input_data)

            # Read the response
            result, resp = win32file.ReadFile(self.handle, 64*1024)
            msg = resp.decode("utf-8")
            print(f"Received: {msg.strip()}")

            return msg
        except pywintypes.error as e:
            print(f"Error during communication: {e}")
            self.disconnect()
            self.connect()  # Try to reconnect
            return None

    def disconnect(self):
        """Close the connection"""
        if self.handle:
            win32file.CloseHandle(self.handle)
            self.handle = None
            self.connected = False
            print("Disconnected from pipe server")

def get_first_double(my_string):
    if my_string is None:
        global none_count
        none_count += 1
        return 0

    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    # Check if there are any matches
    matches = rx.findall(my_string)
    if not matches:
        # No numeric values found
        none_count += 1
        return 0
    var = matches[0]
    return float(var)

# Create a client
client = PipeClient()

# Connect once at the beginning
while not client.connect():
    print("Trying to connect...")
    time.sleep(1)

# Main test with your example commands
try:
    print("\n--- Running example commands ---")

    print("\nSetting objective position 1...")
    client.send_command('SETOBJ1')

    print("\nTaking picture...")
    client.send_command('PIC:C://Users//Feitze//test')

    print("\nGetting current objective position...")
    client.send_command('GETOBJ')

    print("\nGetting magnification...")
    mag = get_first_double(client.send_command('GETMAG'))
    print(f"Magnification: {mag}x")

    print("\nGetting X position...")
    pos_x = get_first_double(client.send_command('GETPOSX'))
    print(f"Position X: {pos_x}")

    print("\nGetting Z position...")
    pos_z = get_first_double(client.send_command('GETZ'))
    print(f"Position Z: {pos_z}")

    print("\nTurning LED on...")
    client.send_command('LEDON')

    print("\nTurning LED off...")
    client.send_command('LEDOFF')

    print("\nGetting F position and adding 0.1mm...")
    pos_f = get_first_double(client.send_command('GETPOSF')) + 0.1
    print(f"New Position F: {pos_f}")

    print("\nSetting new F position...")
    client.send_command(f'SETPOSF{pos_f}')

    print("\nSetting F value...")
    client.send_command(f'SETF{pos_f}')

except KeyboardInterrupt:
    print("Operation interrupted by user")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    # Make sure to disconnect when done
    client.disconnect()

    # Calculate and print execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Number of None/empty responses: {none_count}")