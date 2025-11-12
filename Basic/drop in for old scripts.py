import time
import sys
import win32pipe, win32file, pywintypes
import re
import atexit

# Record start time
start_time = time.time()

# Counter for None/empty responses
none_count = 0

# Create a singleton PipeClient instance
_pipe_client = None

def _get_pipe_client():
    """Singleton getter for the pipe client instance"""
    global _pipe_client
    if _pipe_client is None:
        _pipe_client = PipeClient()
        # Attempt to connect immediately
        _pipe_client.connect()
        # Register disconnect on program exit
        atexit.register(_pipe_client.disconnect)
    return _pipe_client

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
            return True
        except pywintypes.error as e:
            if e.args[0] == 2:
                # No pipe found, similar to old behavior
                self.handle = None
                self.connected = False
            else:
                # Other error
                self.handle = None
                self.connected = False
            return False

    def send_command(self, command):
        """Send a command and receive response"""
        if not self.connected:
            if not self.connect():
                return None

        try:
            # Send the command
            input_data = str.encode(command + '\n')
            win32file.WriteFile(self.handle, input_data)

            # Read the response
            result, resp = win32file.ReadFile(self.handle, 64*1024)
            msg = resp.decode("utf-8")
            return msg
        except pywintypes.error as e:
            # Try to reconnect on error
            self.disconnect()
            self.connect()
            return None

    def disconnect(self):
        """Close the connection"""
        if self.handle:
            try:
                win32file.CloseHandle(self.handle)
            except:
                pass
            self.handle = None
            self.connected = False

# Drop-in replacement for the old Send function
def Send(data):
    global none_count
    client = _get_pipe_client()

    # Try to send up to 3 times (matching original behavior)
    for attempt in range(3):
        try:
            if not client.connected:
                if not client.connect():
                    print("no pipe, trying again in a sec")
                    time.sleep(1)
                    continue

            msg = client.send_command(data)

            if msg is None:
                continue

            print(f"message: {msg}")

            # Check if response is empty
            if not msg or msg.strip() == "":
                none_count += 1

            return msg

        except pywintypes.error as e:
            if e.args[0] == 2:
                print("no pipe, trying again in a sec")
                time.sleep(1)
            elif e.args[0] == 109:
                print("broken pipe, bye bye")
                none_count += 1
                return ""

    # If we get here, all attempts failed
    none_count += 1
    return None

def get_first_double(my_string):
    print(my_string)
    if (my_string is None):
        # This will also increment none_count
        global none_count
        none_count += 1
        return 0;
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
# Rest of the code, for example
for i in range(100000):
    print(get_first_double(Send('GETMAG')))
    print(get_first_double(Send('GETPOSX')))